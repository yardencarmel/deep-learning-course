# test_all_models.py
import torch
from main import LeNet5, get_data_loaders, evaluate, device


def test_all_models():
    """Test all saved models"""

    # Configuration mapping: (name, use_dropout, use_batch_norm)
    configurations = [
        ('Baseline', False, False),
        ('Dropout', True, False),
        ('Weight_Decay', False, False),
        ('Batch_Normalization', False, True),
    ]

    # Get test data once (shared for all models)
    _, test_loader = get_data_loaders()

    results = []

    for config_name, use_dropout, use_batch_norm in configurations:
        print(f"\n{'=' * 70}")
        print(f"Testing: {config_name}")
        print(f"{'=' * 70}")

        for model_type in ['best', 'final']:
            file_path = f'models/{config_name}_{model_type}.pth'

            try:
                # Create model
                model = LeNet5(use_dropout=use_dropout, use_batch_norm=use_batch_norm).to(device)

                # Load weights
                model.load_state_dict(torch.load(file_path, map_location=device))

                # Evaluate
                test_acc = evaluate(model, test_loader, training_mode=False)

                results.append((config_name, model_type, test_acc))
                print(f"  {model_type:>5s}: {test_acc:>6.2f}%")

            except FileNotFoundError:
                print(f"  {model_type:>5s}: File not found")
            except Exception as e:
                print(f"  {model_type:>5s}: Error - {e}")

    # Print summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Configuration':<25} {'Type':<10} {'Test Accuracy':<15}")
    print("-" * 70)
    for config_name, model_type, test_acc in results:
        print(f"{config_name:<25} {model_type:<10} {test_acc:>13.2f}%")
    print("=" * 70)


if __name__ == '__main__':
    test_all_models()