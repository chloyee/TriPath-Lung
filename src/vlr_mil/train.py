"""Patient-stratified internal five-fold training; no external data access."""
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from .data import ClinicalTransform, five_folds, load_patient, read_manifest, CONCEPTS
from .losses import ordinal_loss, ranking_loss, channel_binding_loss
from .metrics import fit_cutpoints, metrics, probabilities
from .model import VLRMIL
from .selection import prepare_bags


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def write_predictions(path, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(embeddings, config, device):
    return VLRMIL(embeddings, adapter_dim=config['adapter_dim'], adapter_dropout=config['adapter_dropout']).to(device)


def predict(model, transform, groups, ids, features_root, device, cutpoints=None):
    model.eval(); rows = []
    with torch.inference_mode():
        for pid in ids:
            r = groups[pid]
            output = model(load_patient(r, features_root, device), *transform(r[0], device))
            score = float(output['risk_score'].cpu())
            p = output['probabilities'].cpu().numpy() if cutpoints is None else probabilities([score], cutpoints)[0]
            rows.append(dict(patient_id=pid, true_label=int(r[0]['label']), predicted_label=int(p.argmax()),
                             prob_non_mpr=float(p[0]), prob_mpr_only=float(p[1]), prob_pcr=float(p[2]),
                             shared_ordinal_score=score))
    return rows


def summarize(rows):
    return metrics([r['true_label'] for r in rows], np.array([[r[k] for k in ['prob_non_mpr', 'prob_mpr_only', 'prob_pcr']] for r in rows]))


def train_fold(args, config, groups, split, fold, embeddings):
    seed_all(config['seed'] + fold)
    device = torch.device(args.device)
    dest = args.output / f"seed{config['seed']}_fold{fold}"
    dest.mkdir(parents=True, exist_ok=False)
    train_ids, valid_ids = split['train'], split['validation']
    assert not set(train_ids) & set(valid_ids)
    transform = ClinicalTransform([groups[p][0] for p in train_ids])
    model = build_model(embeddings, config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    y = np.array([int(groups[p][0]['label']) for p in train_ids])
    counts = np.bincount(y, minlength=3)
    class_weights = torch.tensor(len(y)/(3*counts), dtype=torch.float32, device=device)
    threshold_weights = torch.tensor([(y < 1).sum()/(y >= 1).sum(), (y < 2).sum()/(y >= 2).sum()], dtype=torch.float32, device=device)
    dump(dest/'split.json', split); dump(dest/'config.json', config)
    best, stale = float('inf'), 0
    binding_weight = float(config.get('channel_binding_weight', 0.))
    warmup = int(config.get('channel_binding_warmup_epochs', 5))
    if binding_weight < 0 or warmup < 0:
        raise ValueError('Binding weight and warmup must be nonnegative')
    for epoch in range(1, config['epochs']+1):
        model.train(); order = list(train_ids)
        mpr_ids = [pid for pid in train_ids if int(groups[pid][0]['label']) == 1]
        extra = round((config['mpr_oversample_factor']-1)*len(mpr_ids))
        order.extend(mpr_ids[i % len(mpr_ids)] for i in range(extra))
        random.shuffle(order)
        memory, total = [], 0.
        binding_total = 0.
        binding_active = binding_weight > 0 and epoch > warmup
        delta_sum = np.zeros((3, 3)); delta_count = np.zeros(3, dtype=int)
        for pid in order:
            r = groups[pid]
            slides = load_patient(r, args.features_root, device, training=True, patch_dropout=config['patch_dropout'])
            forward = model.forward_with_channel_ablation if binding_active else model
            output = forward(slides, *transform(r[0], device))
            label = torch.tensor(int(r[0]['label']), device=device)
            loss = ordinal_loss(output, label, class_weights, threshold_weights)
            if binding_active:
                auxiliary = channel_binding_loss(output, label,
                    config.get('channel_binding_support_margin', .01),
                    config.get('channel_binding_specificity_margin', .01))
                loss = loss + binding_weight * auxiliary
                binding_total += float(auxiliary.detach())
                delta_sum[int(label)] += (output['probabilities'][int(label)] -
                    output['channel_ablated_probabilities'][:, int(label)]).detach().cpu().numpy()
                delta_count[int(label)] += 1
            pair = ranking_loss(output['risk_score'], int(label), memory, config['pairwise_max_pairs'], config['pairwise_mpr_factor'])
            if pair is not None:
                loss = loss + config['pairwise_ranking_weight']*pair
            memory.append((int(label), output['risk_score'].detach()))
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step(); total += float(loss.detach())
        model.eval(); valid_loss = 0.
        with torch.inference_mode():
            for pid in valid_ids:
                r = groups[pid]
                output = model(load_patient(r, args.features_root, device), *transform(r[0], device))
                valid_loss += float(ordinal_loss(output, torch.tensor(int(r[0]['label']), device=device), class_weights, threshold_weights))
        valid_loss /= len(valid_ids)
        predictions = predict(model, transform, groups, valid_ids, args.features_root, device)
        record = dict(epoch=epoch, train_loss=total/len(order), validation_loss=valid_loss, validation=summarize(predictions))
        record.update(channel_binding_active=binding_active, channel_binding_loss=binding_total/len(order),
                      channel_binding_delta_by_true_label=[(delta_sum[k]/delta_count[k]).tolist()
                          if delta_count[k] else None for k in range(3)])
        with (dest/'history.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(record)+'\n')
        print(json.dumps(dict(fold=fold, **record)), flush=True)
        if not np.isfinite(valid_loss):
            raise RuntimeError('Nonfinite validation loss')
        if valid_loss < best-config['min_delta']:
            best, stale = valid_loss, 0
            torch.save(dict(model=model.state_dict(), clinical=transform.state, config=config, epoch=epoch,
                            validation_loss=best, split=split, internal_patient_ids=sorted(groups),
                            internal_center=config['internal_center'], concept_names=CONCEPTS), dest/'best.pt')
        else:
            stale += 1
            if stale >= config['patience']:
                break
    checkpoint = torch.load(dest/'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    if config['calibrate_cumulative']:
        train_predictions = predict(model, transform, groups, train_ids, args.features_root, device)
        checkpoint['calibrated_cutpoints'] = fit_cutpoints(
            [r['true_label'] for r in train_predictions], [r['shared_ordinal_score'] for r in train_predictions])
        torch.save(checkpoint, dest/'best.pt')
    rows = predict(model, transform, groups, valid_ids, args.features_root, device, checkpoint.get('calibrated_cutpoints'))
    write_predictions(dest/'validation_predictions.csv', rows)
    dump(dest/'metrics.json', summarize(rows))
    return [dict(fold=fold, **r) for r in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/main.json'))
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--features-root', type=Path, required=True)
    parser.add_argument('--concept-embeddings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--fold', type=int, choices=range(5))
    parser.add_argument('--selection-output', type=Path,
                        help='Build concept-guided bags in a fresh cache directory before training')
    parser.add_argument('--selection-config', type=Path, default=Path('configs/selection.json'))
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    config = json.loads(args.config.read_text())
    groups = read_manifest(args.manifest, internal_center=config['internal_center'])
    splits = five_folds(groups, config['seed'])
    with np.load(args.concept_embeddings, allow_pickle=False) as cache:
        embeddings = torch.from_numpy(cache['embeddings'].astype(np.float32))
        if 'concept_names' in cache and cache['concept_names'].tolist() != CONCEPTS:
            raise ValueError('Incorrect text concept ordering')
    folds = list(range(5)) if args.fold is None else [args.fold]
    for fold in folds:
        if (args.output/f"seed{config['seed']}_fold{fold}").exists():
            raise FileExistsError('Refusing to overwrite an existing fold; use a fresh output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    selection = None
    if args.selection_output is not None:
        selection = prepare_bags(groups, args.features_root, args.selection_output,
                                 json.loads(args.selection_config.read_text()))
        args.features_root = args.selection_output
    dump(args.output/'input_provenance.json', dict(
        manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        concept_embeddings_sha256=hashlib.sha256(args.concept_embeddings.read_bytes()).hexdigest(),
        config=config, selection=selection, features_root=str(args.features_root.resolve()),
        patients=len(groups), folds=folds, torch_version=torch.__version__, numpy_version=np.__version__,
        code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}))
    all_rows = []
    for fold in folds:
        all_rows.extend(train_fold(args, config, groups, splits[fold], fold, embeddings))
    if len(folds) == 5:
        assert len(all_rows) == len({r['patient_id'] for r in all_rows}) == len(groups)
        write_predictions(args.output/'oof_predictions.csv', all_rows)
        dump(args.output/'oof_metrics.json', summarize(all_rows))


if __name__ == '__main__':
    main()
