"""Consolidate all trained models to outputs/models/"""
import shutil
from pathlib import Path
from collections import defaultdict

# Source directories
sources = [
    Path('backend/analysis/outputs/models'),
    Path('inputs/outputs/models'),
    Path('outputs/models'),
]

# Destination
dest = Path('outputs/models')
dest.mkdir(parents=True, exist_ok=True)

# Find all .pkl files
all_models = {}
for source in sources:
    if source.exists():
        for pkl_file in source.rglob('*.pkl'):
            # Get category from parent dir
            category = pkl_file.parent.name if pkl_file.parent.name != 'models' else 'misc'
            model_name = pkl_file.name
            key = f"{category}/{model_name}"

            if key not in all_models:
                all_models[key] = pkl_file

# Copy to destination
categories = defaultdict(list)
for key, source_file in all_models.items():
    category, model_name = key.split('/')
    dest_dir = dest / category
    dest_dir.mkdir(exist_ok=True)
    dest_file = dest_dir / model_name
    shutil.copy2(source_file, dest_file)
    categories[category].append(model_name)

# Report
print("="*80)
print("MODELS CONSOLIDATED")
print("="*80)
print(f"\nTotal unique models: {len(all_models)}\n")

for category in sorted(categories.keys()):
    models = sorted(categories[category])
    print(f"{category}: {len(models)} models")
    for model in models:
        print(f"  • {model}")
    print()

# Market count
market_count = {
    'comprehensive': 12,
    'derivative': 10,
    'kicker': 3,
    'td_scorer': 3,
    'combo': 1,
    'pbp': 5,
    'game_outcome': 3,
}

total_markets = sum(market_count.get(cat, len(models)) for cat, models in categories.items())
print("="*80)
print(f"TOTAL MARKETS TRAINED: {total_markets}/80 ({total_markets/80*100:.1f}%)")
print("="*80)
