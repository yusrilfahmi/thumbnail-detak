import requests
import os

# --- KONFIGURASI ---
base_url = "https://detak.media/wp-json/wp/v2/tags"
checkpoint_file = "last_tag_id.txt"  # File penyimpan ID terakhir (untuk logika sistem)
output_file = "tags.txt"             # File hasil text (untuk dibaca manusia)

# 1. Baca ID terakhir dari file (kalau ada)
last_known_id = 0
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, "r") as f:
        content = f.read().strip()
        if content.isdigit():
            last_known_id = int(content)

print(f"ID Terakhir di sistem: {last_known_id}")
print("Mencari tags baru...")

new_tags = []
page = 1
keep_fetching = True
highest_id_in_session = last_known_id 

while keep_fetching:
    try:
        # Request dengan urutan dari ID terbaru
        params = {
            'per_page': 100,
            'page': page,
            'orderby': 'id',
            'order': 'desc' 
        }
        
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(base_url, params=params, headers=headers)

        if response.status_code == 200:
            data = response.json()
            
            if len(data) == 0:
                break 

            for item in data:
                current_id = item['id']
                
                # Update calon ID tertinggi untuk checkpoint
                if current_id > highest_id_in_session:
                    highest_id_in_session = current_id

                # Cek apakah ini barang baru?
                if current_id > last_known_id:
                    # Masukkan ke list
                    new_tags.append(item['name'])
                else:
                    # Ketemu data lama, stop total
                    keep_fetching = False
                    break 
            
            if keep_fetching:
                print(f"Halaman {page} diperiksa...")
                page += 1
            else:
                print("Batas data lama ditemukan. Berhenti fetching.")
                
        else:
            print(f"Gagal akses API. Status: {response.status_code}")
            break

    except Exception as e:
        print(f"Error: {e}")
        break

# --- PENYIMPANAN KE FILE TXT ---

if new_tags:
    print(f"\nDitemukan {len(new_tags)} tags BARU.")
    
    # Simpan nama tags ke file tags.txt
    # Gunakan 'a' (append) untuk menambah ke bawah
    # encoding='utf-8' penting supaya karakter aneh/emoji tidak error
    try:
        with open(output_file, "a", encoding="utf-8") as f:
            for tag in new_tags:
                f.write(tag + "\n")
        
        print(f"Sukses! List tag baru telah ditambahkan ke '{output_file}'")
        
        # Update checkpoint ID hanya jika penyimpanan file sukses
        with open(checkpoint_file, "w") as f:
            f.write(str(highest_id_in_session))
            
    except Exception as e:
        print(f"Gagal menulis file: {e}")

else:
    print("\nTidak ada tags baru untuk disimpan.")