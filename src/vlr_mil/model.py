"""Main VLR-MIL architecture operating on frozen patch and concept features."""
import torch
from torch import nn
from torch.nn import functional as F


def percentile_rank_columns(values):
    if values.ndim != 2:
        raise ValueError('Expected patches x concepts')
    if len(values) <= 1:
        return torch.full_like(values, .5)
    order = values.argsort(dim=0, stable=True)
    ranks = torch.empty_like(order)
    ranks.scatter_(0, order, torch.arange(len(values), device=values.device)[:, None].expand_as(order))
    return ranks.to(values.dtype) / (len(values) - 1)


class ClassSpecificAttention(nn.Module):
    """Three gated pooling channels, normalized over instances independently."""
    def __init__(self, dim, classes=3, dropout=.3):
        super().__init__()
        self.value = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Dropout(dropout))
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid(), nn.Dropout(dropout))
        self.attention = nn.Linear(dim, classes, bias=False)
        self.output = nn.Linear(dim * classes, dim)

    def forward(self, inputs, ablate_channel=None):
        logits = self.attention(self.value(inputs) * self.gate(inputs)).transpose(0, 1)
        weights = logits.softmax(dim=-1)
        pooled = weights @ inputs
        if ablate_channel is not None:
            if ablate_channel not in range(len(pooled)):
                raise ValueError('Invalid attention channel')
            mask = pooled.new_ones((len(pooled), 1))
            mask[ablate_channel] = 0
            pooled = pooled * mask
        return self.output(pooled.flatten()), weights


class FeatureAdapter(nn.Module):
    def __init__(self, feature_dim=512, bottleneck=64, dropout=.1):
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.down = nn.Linear(feature_dim, bottleneck)
        self.up = nn.Linear(bottleneck, feature_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, features):
        return features + self.up(self.dropout(F.gelu(self.down(self.norm(features)))))


class SpatialSlideEncoder(nn.Module):
    def __init__(self, feature_dim=512, concept_dim=8, dim=128):
        super().__init__()
        self.feature = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, dim))
        self.matched_text = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, dim))
        self.concepts = nn.Sequential(nn.LayerNorm(concept_dim), nn.Linear(concept_dim, dim))
        self.coordinates = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.alignment_gate = nn.Sequential(nn.Linear(dim * 2 + 1, dim), nn.GELU(),
                                           nn.Dropout(.3), nn.Linear(dim, dim), nn.Sigmoid())
        self.feature_dropout = nn.Dropout(.25)
        layer = nn.TransformerEncoderLayer(dim, 4, dim * 2, .3, activation='gelu',
                                           batch_first=True, norm_first=True)
        self.context = nn.TransformerEncoder(layer, num_layers=1)
        self.pool = ClassSpecificAttention(dim)

    def forward(self, features, concepts, coordinates, matched_text, alignment_score, ablate_channel=None):
        centered = coordinates - coordinates.mean(dim=0, keepdim=True)
        positions = centered / centered.abs().amax(dim=0, keepdim=True).clamp_min(1.)
        image = self.feature_dropout(self.feature(features))
        text = self.feature_dropout(self.matched_text(matched_text))
        gate = self.alignment_gate(torch.cat((image, text, alignment_score.unsqueeze(-1)), dim=-1))
        tokens = gate * image + (1 - gate) * text + self.concepts(concepts) + self.coordinates(positions)
        return self.pool(self.context(tokens.unsqueeze(0)).squeeze(0), ablate_channel)


class ClinicalEncoder(nn.Module):
    """Histology, T stage, smoking; age/TPS features and five missing flags."""
    def __init__(self, dim=128):
        super().__init__()
        self.embeddings = nn.ModuleList(nn.Embedding(size, 8) for size in (4, 6, 4))
        self.network = nn.Sequential(nn.Linear(33, dim), nn.GELU(), nn.Dropout(.4),
                                     nn.Linear(dim, dim), nn.LayerNorm(dim))

    def forward(self, categorical, continuous, missing):
        embedded = [layer(categorical[i]) for i, layer in enumerate(self.embeddings)]
        return self.network(torch.cat([*embedded, continuous.reshape(-1), missing.float()]))


class CumulativeLinkHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1)
        self.cutpoint1 = nn.Parameter(torch.tensor(0.))
        self.cutpoint_gap_raw = nn.Parameter(torch.tensor(.5))

    def cutpoints(self):
        return torch.stack((self.cutpoint1, self.cutpoint1 + F.softplus(self.cutpoint_gap_raw) + 1e-3))

    def from_score(self, score):
        score = score.reshape(())
        cuts = self.cutpoints()
        q1, q2 = torch.sigmoid(score - cuts).unbind()
        return dict(raw_risk_score=score, risk_score=score, cutpoints=cuts,
                    q1=q1, q2=q2, probabilities=torch.stack((1-q1, q1-q2, q2)))


class VLRMIL(nn.Module):
    """Full adapter + concept-percentile + residual-clinical cumulative model."""
    def __init__(self, concept_embeddings, feature_dim=512, concept_dim=8, dim=128,
                 adapter_dim=64, adapter_dropout=.1, slide_dropout=.15, chunk_size=512):
        super().__init__()
        if tuple(concept_embeddings.shape) != (concept_dim, feature_dim):
            raise ValueError('Concept embeddings must have shape (8, feature_dim)')
        self.chunk_size = chunk_size
        self.feature_adapter = FeatureAdapter(feature_dim, adapter_dim, adapter_dropout)
        self.slide_encoder = SpatialSlideEncoder(feature_dim, concept_dim, dim)
        self.chunk_pool = ClassSpecificAttention(dim)
        self.slide_pool = ClassSpecificAttention(dim)
        self.clinical_encoder = ClinicalEncoder(dim)
        # Retained for state-dict and initialization compatibility; inactive in residual fusion.
        self.gate = nn.Sequential(nn.Linear(dim*2, dim), nn.GELU(), nn.Dropout(.3), nn.Linear(dim, dim), nn.Sigmoid())
        self.head = CumulativeLinkHead(dim)
        self.image_score = nn.Linear(dim, 1)
        self.clinical_score = nn.Linear(dim, 1)
        self.interaction = nn.Sequential(nn.LayerNorm(dim*3), nn.Linear(dim*3, dim),
                                         nn.GELU(), nn.Dropout(.4), nn.Linear(dim, 3))
        nn.init.zeros_(self.interaction[-1].weight)
        nn.init.zeros_(self.interaction[-1].bias)
        self.clinical_scale_logit = nn.Parameter(torch.tensor(-2.2))
        self.interaction_scale_logit = nn.Parameter(torch.tensor(-3.))
        self.slide_dropout = slide_dropout
        self.register_buffer('concept_embeddings', concept_embeddings.float())

    def encode_slide(self, slide, ablate_patch_channel=None):
        embeddings, attentions = [], []
        for start in range(0, len(slide['features']), self.chunk_size):
            stop = start + self.chunk_size
            raw = slide['concept_scores'][start:stop]
            concepts = percentile_rank_columns(raw)
            embedding, attention = self.slide_encoder(
                self.feature_adapter(slide['features'][start:stop]), concepts,
                slide['coordinates'][start:stop], self.concept_embeddings[raw.argmax(dim=-1)],
                concepts.amax(dim=-1), ablate_patch_channel)
            embeddings.append(embedding)
            attentions.append(attention)
        embedding = embeddings[0] if len(embeddings) == 1 else self.chunk_pool(torch.stack(embeddings))[0]
        return embedding, attentions

    def forward(self, slides, clinical_categorical, clinical_age, clinical_missing, ablate_patch_channel=None):
        clinical = self.clinical_encoder(clinical_categorical, clinical_age, clinical_missing)
        if self.training and len(slides) > 1 and self.slide_dropout > 0:
            keep = torch.rand(len(slides), device=slides[0]['features'].device) >= self.slide_dropout
            if not keep.any():
                keep[torch.randint(len(slides), (1,), device=keep.device)] = True
            slides = [s for i, s in enumerate(slides) if bool(keep[i])]
        encoded = [self.encode_slide(s, ablate_patch_channel) for s in slides]
        image, slide_attention = self.slide_pool(torch.stack([x[0] for x in encoded]))
        clinical_scale = self.clinical_scale_logit.sigmoid()
        interaction_scale = self.interaction_scale_logit.sigmoid()
        interaction = self.interaction(torch.cat((image, clinical, image * clinical)))
        image_score = self.image_score(image).reshape(())
        clinical_score = self.clinical_score(clinical).reshape(())
        result = self.head.from_score(image_score + clinical_scale * clinical_score + interaction_scale * interaction.mean())
        result.update(image_score=image_score, clinical_score=clinical_score,
                      clinical_scale=clinical_scale, interaction_scale=interaction_scale,
                      image_gate=1-clinical_scale, clinical_gate=clinical_scale,
                      patch_attention=[x[1] for x in encoded], slide_attention=slide_attention,
                      patient_embedding=image + clinical_scale * clinical)
        return result

    def forward_with_channel_ablation(self, slides, *clinical):
        """Same head, same stochastic masks; zero one PATCH-pool vector per pass.

        Channel indices refer only to the patch pool, not chunk/slide pooling.
        The outer RNG advances by one ordinary forward, even during training.
        """
        device = slides[0]['features'].device
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
        cpu_state = torch.get_rng_state()
        cuda_states = [torch.cuda.get_rng_state(d) for d in devices]
        result = self(slides, *clinical)
        ablated = []
        for channel in range(3):
            with torch.random.fork_rng(devices=devices):
                torch.set_rng_state(cpu_state)
                for d, state in zip(devices, cuda_states):
                    torch.cuda.set_rng_state(state, d)
                ablated.append(self(slides, *clinical, ablate_patch_channel=channel)['probabilities'])
        result['channel_ablated_probabilities'] = torch.stack(ablated)
        return result
