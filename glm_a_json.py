#!/usr/bin/env python3
"""Baja los rayos reales del GOES-19 (GLM, NOAA) y arma rayos.json para la app.
Uso:  pip install netCDF4 numpy ; python glm_a_json.py salida/rayos.json
Fuente: bucket público de NOAA (sin clave). Zona: sur de Sudamérica (Mendoza y alrededores).
NO PROBADO con datos reales todavía: ejecutarlo una vez a mano y revisar el resultado."""
import sys, os, re, json, time, io, urllib.request, xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
BUCKET = os.environ.get('GLM_BUCKET', 'noaa-goes19')
LAT = (-36.0, -29.5); LON = (-72.5, -65.0); MINUTOS = 60
EPOCH_GLM = 946728000  # 2000-01-01 12:00:00 UTC

def get(url):
    with urllib.request.urlopen(url, timeout=40) as r: return r.read()

def listar(hora):
    pre = f"GLM-L2-LCFA/{hora:%Y}/{hora:%j}/{hora:%H}/"
    xml = get(f"https://{BUCKET}.s3.amazonaws.com/?list-type=2&prefix={pre}")
    ns = {'s': 'http://s3.amazonaws.com/doc/2006-03-01/'}
    return [k.text for k in ET.fromstring(xml).findall('.//s:Key', ns)]

def inicio(key):
    m = re.search(r'_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})', key)
    y, d, h, mi, s = map(int, m.groups())
    return datetime(y, 1, 1, tzinfo=timezone.utc) + timedelta(days=d - 1, hours=h, minutes=mi, seconds=s)

def leer(key, salida, desde):
    import netCDF4, numpy as np
    ds = netCDF4.Dataset('inmemory.nc', memory=get(f"https://{BUCKET}.s3.amazonaws.com/{key}"))
    la = np.ma.filled(ds['flash_lat'][:], np.nan); lo = np.ma.filled(ds['flash_lon'][:], np.nan)
    off = np.ma.filled(ds['flash_time_offset_of_first_event'][:], 0)
    base = float(ds['product_time'][:]) + EPOCH_GLM
    for a, o, t in zip(la, lo, off):
        if np.isnan(a) or np.isnan(o): continue
        ts = base + float(t)
        if LAT[0] <= a <= LAT[1] and LON[0] <= o <= LON[1] and ts >= desde:
            salida.append([round(float(a), 2), round(float(o), 2), int(ts)])
    ds.close()

def main(ruta):
    ahora = datetime.now(timezone.utc); desde = ahora.timestamp() - MINUTOS * 60
    keys = []; fallos = 0
    for h in (ahora - timedelta(hours=1), ahora):
        try: keys += listar(h)
        except Exception as e: fallos += 1; print('lista falló', h, e, file=sys.stderr)
    keys = [k for k in keys if inicio(k).timestamp() >= desde - 60]
    if fallos and not keys: sys.exit('No pude listar archivos de NOAA: no publico para no mostrar un "sin rayos" falso')
    fl = []; malos = 0
    for k in keys:
        try: leer(k, fl, desde)
        except Exception as e: malos += 1; print('archivo falló', k, e, file=sys.stderr)
    if keys and malos == len(keys): sys.exit('Ningún archivo se pudo leer: no publico')
    fl.sort(key=lambda f: f[2])
    os.makedirs(os.path.dirname(ruta) or '.', exist_ok=True)
    json.dump({'t': int(time.time() * 1000), 'src': 'NOAA GOES-19 GLM', 'flashes': fl}, open(ruta, 'w'), separators=(',', ':'))
    print(len(keys), 'archivos,', len(fl), 'descargas')

if __name__ == '__main__': main(sys.argv[1] if len(sys.argv) > 1 else 'rayos.json')
