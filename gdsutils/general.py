import tarfile
import urllib.request
from pathlib import Path


def descargar_datos(url):
    nombre_archivo = url.split("/")[-1]
    folder_name = nombre_archivo.split(".")[0]

    base_dir = Path("data")
    target_dir = base_dir / folder_name

    if target_dir.exists():
        print(f"La carpeta '{target_dir}' ya existe.")
        return

    try:
        base_dir.mkdir(parents=True, exist_ok=True)

        temp_file, _ = urllib.request.urlretrieve(url)

        with tarfile.open(temp_file, "r:gz") as tar:
            tar.extractall(path=base_dir)

    except Exception as e:
        print(f"❌ Error: {e}")

    finally:
        # Limpiar el archivo temporal descargado
        urllib.request.urlcleanup()
