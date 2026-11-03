"""Evaluate a trained TriPath-Lung checkpoint on a held-out patient cohort."""
import argparse
import hashlib
from pathlib import Path

import torch

from .data import ClinicalTransform, CONCEPTS, read_manifest
from .train import build_model, dump, predict, summarize, write_predictions


def evaluate(checkpoint_path, manifest, features_root, output, device='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if checkpoint.get('model_format_version') != 2 or checkpoint.get('concept_names') != CONCEPTS:
        raise ValueError('Expected a TriPath-Lung format-2 checkpoint with the canonical concepts')
    groups = read_manifest(manifest)
    development_ids = set(checkpoint['split']['train']) | set(checkpoint['split']['validation'])
    if development_ids.intersection(groups):
        raise ValueError('Test patients must be separate from training and model-selection patients')
    device = torch.device(device)
    model = build_model(checkpoint['model']['concept_embeddings'], checkpoint['config'], device)
    model.load_state_dict(checkpoint['model'], strict=True)
    transform = ClinicalTransform(state=checkpoint['clinical'])
    rows = predict(model, transform, groups, list(groups), features_root, device)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_predictions(output/'test_predictions.csv', rows)
    dump(output/'metrics.json', summarize(rows))
    dump(output/'test_provenance.json', dict(
        checkpoint_sha256=hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
        selected_epoch=checkpoint['epoch'], patients=len(groups),
        features_root=str(Path(features_root).resolve()),
        clinical=checkpoint['clinical'], concept_names=CONCEPTS))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--features-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    evaluate(args.checkpoint, args.manifest, args.features_root, args.output, args.device)


if __name__ == '__main__':
    main()
