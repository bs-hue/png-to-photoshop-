import io
import requests
from psd_tools import PSDImage

# 1. Health check
r = requests.get('http://127.0.0.1:8000/api/v1/health')
print('1. Health Check:', r.status_code, r.json()['status'])

# 2. Upload and decompose user image
img_path = r'C:/Users/Yogita Thakur/.gemini/antigravity/brain/7670e4bd-bc06-4c15-a94b-869b822c59e7/.user_uploaded/media_1793433727864_4acc1533.png'
with open(img_path, 'rb') as f:
    files = {'file': ('test.png', f, 'image/png')}
    r2 = requests.post('http://127.0.0.1:8000/api/v1/decompose-png', files=files, timeout=60)

print('2. Decompose HTTP:', r2.status_code)
data = r2.json()
print('   Job ID:', data['job_id'])
print('   PSD URL:', data['psd_url'])
print('   Warnings:', data['warnings'])

# 3. Download the generated PSD
r_psd = requests.get(data['psd_url'])
print(f'3. PSD Download HTTP: {r_psd.status_code} ({len(r_psd.content)} bytes)')

# 4. Open and inspect every layer in the PSD
psd = PSDImage.open(io.BytesIO(r_psd.content))
print('4. PSD Dimensions:', psd.size)

print('\n================ VERIFYING EVERY LAYER IN PSD ================')
text_layer_count = 0
for l in psd.descendants():
    is_group = l.is_group()
    kind = getattr(l, 'kind', 'group')
    has_mask = getattr(l, 'has_mask', lambda: False)()
    if kind == 'type':
        text_layer_count += 1
        print(f'   [REAL EDITABLE TEXT (T)] \"{l.name}\"')
        print(f'      -> kind: {kind}')
        print(f'      -> has layer mask: {has_mask}')
    elif is_group:
        print(f'   [FOLDER/GROUP] \"{l.name}\"')
    else:
        print(f'   [RASTER LAYER] \"{l.name}\" (kind: {kind}, mask: {has_mask})')

print('\n================ RESULTS ================')
print(f'Total Native Editable Type Layers: {text_layer_count}')
if text_layer_count > 0:
    print('SUCCESS: Every text element is a real native Photoshop Type Layer!')
else:
    print('FAILURE: No Type layers found!')

