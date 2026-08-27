import pathlib, json, shutil

base = pathlib.Path(r'C:\Users\Administrator\Desktop\CNN\data\processed\val\tamil_brahmi\general_brahmi_letters')
lex_path = pathlib.Path(r'C:\Users\Administrator\Desktop\CNN\data\lexicon.json')
lex = json.loads(lex_path.read_text(encoding='utf-8'))

# Build mapping: file -> transliteration
file_to_trans = {v['reference_file']: v['transliteration'] for k,v in lex['tamil_brahmi_letters'].items()}
print('Mapping:')
for f, t in sorted(file_to_trans.items()):
    print(f' {f} -> {t}')

print('\n--- Current files in root ---')
for f in sorted(base.glob('Screenshot*.png')):
    print(f.name)

print('\n--- Existing subdirs ---')
for d in sorted(base.glob('*')):
    if d.is_dir():
        files = list(d.glob('*.png'))
        print(f'{d.name}: {len(files)} files - {[f.name for f in files]}')

# Move misplaced files back to root
misplaced = [
    (base / 'ai' / 'Screenshot 2026-08-18 010018.png', base / 'Screenshot 2026-08-18 010018.png'),
    (base / 'ha' / 'Screenshot 2026-08-18 010238.png', base / 'Screenshot 2026-08-18 010238.png'),
    (base / 'sa' / 'Screenshot 2026-08-18 010230.png', base / 'Screenshot 2026-08-18 010230.png'),
]
for src, dst in misplaced:
    if src.exists():
        print(f'Moving misplaced {src} -> {dst}')
        shutil.move(str(src), str(dst))

print('\n--- After moving misplaced back, root files ---')
for f in sorted(base.glob('Screenshot*.png')):
    print(f.name)

# Create all needed subdirectories based on lexicon transliterations
unique_trans = sorted(set(file_to_trans.values()))
print(f'\nUnique transliterations: {unique_trans}')
# Also add dha as alias for ja (since image shows dha, lexicon has ja)
if 'dha' not in unique_trans:
    unique_trans.append('dha')
    print('Added dha as alias')

for trans in unique_trans:
    subdir = base / trans
    subdir.mkdir(exist_ok=True)
    print(f'Ensured subdir: {trans}')

# Move each file from root to its correct subdir
for file, trans in file_to_trans.items():
    src = base / file
    if src.exists() and src.is_file():
        dst = base / trans / file
        if dst.exists():
            print(f'Skipping {file} already in {trans}')
            continue
        print(f'Moving {file} -> {trans}/')
        shutil.move(str(src), str(dst))
    else:
        # File already in subdir? Check if it's in correct subdir
        found = False
        for sub in base.glob('*'):
            if sub.is_dir():
                if (sub / file).exists():
                    # Check if it's in correct folder
                    if sub.name != trans:
                        print(f'Found {file} in wrong folder {sub.name}, should be {trans} - moving')
                        shutil.move(str(sub / file), str(base / trans / file))
                    else:
                        print(f'{file} already correctly in {trans}/')
                    found = True
                    break
        if not found and not (base / trans / file).exists():
            print(f'WARNING: {file} not found anywhere!')

# For dha alias: copy ja file to dha
ja_file = base / 'ja' / 'Screenshot 2026-08-18 010245.png'
dha_file = base / 'dha' / 'Screenshot 2026-08-18 010245.png'
if ja_file.exists() and not dha_file.exists():
    print(f'Copying ja file to dha alias: {ja_file} -> {dha_file}')
    shutil.copy(str(ja_file), str(dha_file))
elif (base / 'dha' / 'Screenshot 2026-08-18 010245.png').exists() and not ja_file.exists():
    print(f'Copying dha file to ja: dha -> ja')
    shutil.copy(str(base / 'dha' / 'Screenshot 2026-08-18 010245.png'), str(ja_file))

print('\n--- Final structure ---')
for d in sorted(base.glob('*')):
    if d.is_dir():
        files = list(d.glob('*.png'))
        print(f'{d.name}: {len(files)} files')
print('Root files remaining:')
for f in base.glob('Screenshot*.png'):
    print(f'  {f.name} (should be 0)')

total = sum(len(list(d.glob('*.png'))) for d in base.glob('*') if d.is_dir())
print(f'\nTotal files in subdirs: {total}')
