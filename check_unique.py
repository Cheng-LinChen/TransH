import pandas as pd

# -----------------------------
# Load KG
# -----------------------------
df = pd.read_csv("kg.csv")

# -----------------------------
# Normalize X and Y entities
# -----------------------------
x = df[['x_id', 'x_name', 'x_type']].rename(
    columns={'x_id': 'id', 'x_name': 'name', 'x_type': 'type'}
)

y = df[['y_id', 'y_name', 'y_type']].rename(
    columns={'y_id': 'id', 'y_name': 'name', 'y_type': 'type'}
)

entities = pd.concat([x, y], ignore_index=True).drop_duplicates()

# -----------------------------
# Rule checks
# -----------------------------

# Rule 1: Same ID → different names
r1 = (
    entities.groupby('id')['name']
    .nunique()
    .reset_index(name='name_count')
    .query('name_count > 1')
)
r1['violation'] = 'Same ID → different names'

# Rule 2: Same ID → different types
r2 = (
    entities.groupby('id')['type']
    .nunique()
    .reset_index(name='type_count')
    .query('type_count > 1')
)
r2['violation'] = 'Same ID → different types'

# Rule 3: Same name → different IDs
r3 = (
    entities.groupby('name')['id']
    .nunique()
    .reset_index(name='id_count')
    .query('id_count > 1')
)
r3['violation'] = 'Same name → different IDs'

# Rule 4: Same name → different types
r4 = (
    entities.groupby('name')['type']
    .nunique()
    .reset_index(name='type_count')
    .query('type_count > 1')
)
r4['violation'] = 'Same name → different types'

# Rule 5: Same (id, name) → different types
r5 = (
    entities
    .groupby(['id', 'name'])['type']
    .nunique()
    .reset_index(name='type_count')
    .query('type_count > 1')
)
r5['violation'] = 'Same (id, name) → different types'

# -----------------------------
# Combine violations
# -----------------------------
violations = pd.concat([r1, r2, r3, r4, r5], ignore_index=True)


# -----------------------------
# Extract problematic rows
# -----------------------------
bad_ids = set(r1['id']).union(r2['id']).union(r5['id'])
bad_names = set(r3['name']).union(r4['name']).union(r5['name'])

problematic_rows = entities[
    (entities['id'].isin(bad_ids)) |
    (entities['name'].isin(bad_names))
].sort_values(['id', 'name'])


# -----------------------------
# Statistics
# -----------------------------
stats = {
    "total_entity_rows": len(entities),
    "unique_ids": entities['id'].nunique(),
    "unique_names": entities['name'].nunique(),
    "violations_total": len(violations),
    "ids_with_issues": len(bad_ids),
    "names_with_issues": len(bad_names),
}


# -----------------------------
# Output
# -----------------------------
print("\n=== KG ENTITY CONSISTENCY STATISTICS ===")
for k, v in stats.items():
    print(f"{k}: {v}")

print("\n=== VIOLATIONS SUMMARY ===")
print(violations)

print("\n=== PROBLEMATIC ENTITY ROWS ===")
print(problematic_rows)


print("\n=== VIOLATION COUNTS BY RULE ===")
print(f"r1 (Same ID → different names): {len(r1)}")
print(f"r2 (Same ID → different types): {len(r2)}")
print(f"r3 (Same name → different IDs): {len(r3)}")
print(f"r4 (Same name → different types): {len(r4)}")
print(f"r5 (Same (id, name) → different types): {len(r5)}")



total_violation_count = len(violations)
print(f"\nTOTAL VIOLATION COUNT (Rules 1-5): {total_violation_count}")
