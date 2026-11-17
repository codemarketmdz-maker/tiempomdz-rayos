#!/usr/bin/env python3
"""Baja la lluvia estimada por el GOES-19 (ABI-L2-RRQPEF, NOAA) y arma lluvia.json para la app.
Uso:  pip install netCDF4 numpy ; python goes_lluvia_json.py salida/lluvia.json
Fuente: bucket público de NOAA (sin clave), igual que glm_a_json.py. Zona: sur de Sudamérica (Mendoza y alrededores).
Formato: {"t":ms de publicación,"obs":epoch seg de la imagen,"b":[sur,oeste,norte,este],"res":grados,
          "cells":[[col,fila,mm_por_hora*10],...]}  col desde el oeste, fila desde el norte; solo celdas con lluvia.
Una imagen cada ~10 minutos. NO PROBADO con datos reales de NOAA: la matemática de la proyección sí se probó con datos
sintéticos (python goes_lluvia_json.py --test); ejecutarlo una vez a mano y revisar el resultado."""
import sys, os, re, json, time, urllib.request, xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
import numpy as np

BUCKET = os.environ.get('GOES_BUCKET', 'noaa-goes19')
PRODUCTO = 'ABI-L2-RRQPEF'
BBOX = (-36.0, -72.5, -29.5, -65.0)   # sur, oeste, norte, este
RES = 0.05                            # grados por celda (~5 km)
MIN_MMH = 0.1                         # menos que esto se considera sin lluvia
MAX_EDAD_MIN = 60                     # si la última imagen es más vieja, no se publica

def get(url):
    with urllib.request.urlopen(url, timeout=60) as r: return r.read()

def listar(hora):
    pre = f"{PRODUCTO}/{hora:%Y}/{hora:%j}/{hora:%H}/"
    xml = get(f"https://{BUCKET}.s3.amazonaws.com/?list-type=2&prefix={pre}")
    ns = {'s': 'http://s3.amazonaws.com/doc/2006-03-01/'}
    return [k.text for k in ET.fromstring(xml).findall('.//s:Key', ns)]

def inicio(key):
    m = re.search(r'_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})', key)
    y, d, h, mi, s = map(int, m.groups())
    return datetime(y, 1, 1, tzinfo=timezone.utc) + timedelta(days=d - 1, hours=h, minutes=mi, seconds=s)

# ---- Proyección geoestacionaria GOES-R (fórmulas de la guía oficial PUG, vol. 3) ----
class Proy:
    def __init__(self, H, req, rpol, lon0):
        self.H = H + req; self.req = req; self.rpol = rpol; self.lon0 = np.radians(lon0)
        self.e2 = (req ** 2 - rpol ** 2) / req ** 2
    def directa(self, lat, lon):
        """lat, lon (grados) -> ángulos de escaneo x, y (radianes)"""
        phi = np.radians(lat); lam = np.radians(lon)
        phic = np.arctan((self.rpol ** 2 / self.req ** 2) * np.tan(phi))
        rc = self.rpol / np.sqrt(1 - self.e2 * np.cos(phic) ** 2)
        sx = self.H - rc * np.cos(phic) * np.cos(lam - self.lon0)
        sy = -rc * np.cos(phic) * np.sin(lam - self.lon0)
        sz = rc * np.sin(phic)
        return np.arcsin(-sy / np.sqrt(sx ** 2 + sy ** 2 + sz ** 2)), np.arctan(sz / sx)
    def inversa(self, x, y):
        """x, y (radianes) -> lat, lon (grados); NaN si el punto mira al espacio"""
        a = np.sin(x) ** 2 + np.cos(x) ** 2 * (np.cos(y) ** 2 + (self.req ** 2 / self.rpol ** 2) * np.sin(y) ** 2)
        b = -2 * self.H * np.cos(x) * np.cos(y)
        c = self.H ** 2 - self.req ** 2
        with np.errstate(invalid='ignore'):
            rs = (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
            sx = rs * np.cos(x) * np.cos(y); sy = -rs * np.sin(x); sz = rs * np.cos(x) * np.sin(y)
            lat = np.degrees(np.arctan((self.req ** 2 / self.rpol ** 2) * sz / np.sqrt((self.H - sx) ** 2 + sy ** 2)))
            lon = np.degrees(self.lon0 - np.arctan(sy / (self.H - sx)))
        return lat, lon

def recortar(x, y, rr, proy, bbox=BBOX, res=RES, minimo=MIN_MMH):
    """x (n,), y (m,) en radianes; rr (m, n) mm/h (NaN o enmascarado = sin dato). Devuelve la lista de celdas con lluvia."""
    s, w, n, e = bbox
    la, lo = np.meshgrid(np.linspace(s, n, 40), np.linspace(w, e, 40))
    bx, by = proy.directa(la, lo)
    mx = 3 * abs(float(np.median(np.diff(x)))); my = 3 * abs(float(np.median(np.diff(y))))
    ix = np.where((x >= bx.min() - mx) & (x <= bx.max() + mx))[0]
    iy = np.where((y >= by.min() - my) & (y <= by.max() + my))[0]
    if not len(ix) or not len(iy): raise ValueError('la zona no cae dentro de la imagen del satélite')
    sub = np.ma.filled(rr[iy[0]:iy[-1] + 1, ix[0]:ix[-1] + 1].astype('float64'), np.nan)
    X, Y = np.meshgrid(x[ix[0]:ix[-1] + 1], y[iy[0]:iy[-1] + 1])
    lat, lon = proy.inversa(X, Y)
    ok = np.isfinite(sub) & np.isfinite(lat) & np.isfinite(lon) & (lat >= s) & (lat < n) & (lon >= w) & (lon < e)
    cols = int(round((e - w) / res)); rows = int(round((n - s) / res))
    grid = np.zeros((rows, cols))
    ci = np.floor((lon[ok] - w) / res).astype(int); ri = np.floor((n - lat[ok]) / res).astype(int)
    np.maximum.at(grid, (np.clip(ri, 0, rows - 1), np.clip(ci, 0, cols - 1)), sub[ok])
    rj, cj = np.where(grid >= minimo)
    return [[int(c), int(r), int(round(grid[r, c] * 10))] for r, c in zip(rj, cj)], (cols, rows), int(np.isfinite(sub).sum())

def leer(key):
    import netCDF4
    ds = netCDF4.Dataset('inmemory.nc', memory=get(f"https://{BUCKET}.s3.amazonaws.com/{key}"))
    p = ds['goes_imager_projection']
    proy = Proy(float(p.perspective_point_height), float(p.semi_major_axis), float(p.semi_minor_axis), float(p.longitude_of_projection_origin))
    x = np.array(ds['x'][:], dtype='float64'); y = np.array(ds['y'][:], dtype='float64')
    rr = ds['RRQPE'][:]
    cells, dim, validos = recortar(x, y, rr, proy)
    ds.close()
    return cells, validos

def prefijos(pre):
    xml = get(f"https://{BUCKET}.s3.amazonaws.com/?list-type=2&delimiter=/&prefix={pre}")
    ns = {'s': 'http://s3.amazonaws.com/doc/2006-03-01/'}
    return [p.text for p in ET.fromstring(xml).findall('.//s:CommonPrefixes/s:Prefix', ns)]

def diagnostico(ahora):
    """Si no hay imágenes, muestra en el log qué hay realmente en el bucket (para ajustar el producto)."""
    print('--- DIAGNÓSTICO ---', file=sys.stderr)
    try:
        ps = prefijos('ABI-L2-')
        print('Productos ABI-L2 en', BUCKET + ':', ', '.join(p.rstrip('/') for p in ps), file=sys.stderr)
        print('Los que parecen de lluvia:', [p for p in ps if 'RR' in p or 'RAIN' in p.upper()], file=sys.stderr)
    except Exception as e: print('No pude listar productos:', e, file=sys.stderr)
    try:
        anios = prefijos(PRODUCTO + '/'); print(PRODUCTO, 'años:', anios, file=sys.stderr)
        if anios:
            dias = prefijos(anios[-1]); print('últimos días:', dias[-3:], file=sys.stderr)
    except Exception as e: print('No pude listar', PRODUCTO, e, file=sys.stderr)
    print('Hora UTC del servidor:', ahora, file=sys.stderr)

def main(ruta):
    ahora = datetime.now(timezone.utc); keys = []; fallos = 0
    for h in (ahora - timedelta(hours=1), ahora):
        try: keys += listar(h)
        except Exception as e: fallos += 1; print('lista falló', h, e, file=sys.stderr)
    if not keys:
        diagnostico(ahora)
        sys.exit('No encontré imágenes de lluvia en NOAA: no publico para no mostrar un "sin lluvia" falso')
    keys.sort(key=inicio); key = keys[-1]; t0 = inicio(key)
    edad = (ahora - t0).total_seconds() / 60
    if edad > MAX_EDAD_MIN: sys.exit(f'La última imagen tiene {edad:.0f} min: no publico')
    cells, validos = leer(key)
    if validos == 0: sys.exit('La imagen no trae datos en la zona: no publico')
    os.makedirs(os.path.dirname(ruta) or '.', exist_ok=True)
    json.dump({'t': int(time.time() * 1000), 'obs': int(t0.timestamp()), 'src': 'NOAA GOES-19 RRQPE', 'b': list(BBOX), 'res': RES, 'cells': cells},
              open(ruta, 'w'), separators=(',', ':'))
    print(key.split('/')[-1], f'{edad:.0f} min,', validos, 'píxeles con dato,', len(cells), 'celdas con lluvia')

def prueba():
    """Chequeo con datos inventados: una lluvia sobre Mendoza tiene que caer en la celda de Mendoza."""
    proy = Proy(35786023.0, 6378137.0, 6356752.31414, -75.2)
    x = np.arange(-0.151844, 0.151844, 56e-6); y = np.arange(0.151844, -0.151844, -56e-6)
    mza = (-32.89, -68.84)
    # ida y vuelta de la proyección
    xx, yy = proy.directa(*mza); la, lo = proy.inversa(xx, yy)
    assert abs(la - mza[0]) < 1e-6 and abs(lo - mza[1]) < 1e-6, (la, lo)
    rr = np.ma.masked_all((len(y), len(x)))
    rr[:] = 0.0
    j = int(np.argmin(abs(y - yy))); i = int(np.argmin(abs(x - xx)))
    rr[j - 3:j + 4, i - 3:i + 4] = 6.0
    cells, (cols, rows), validos = recortar(x, y, rr, proy)
    assert cells, 'no salió ninguna celda'
    s, w, n, e = BBOX
    cs = [(w + (c + .5) * RES, n - (r + .5) * RES, v) for c, r, v in cells]
    clon = np.mean([c[0] for c in cs]); clat = np.mean([c[1] for c in cs])
    print('celdas:', len(cells), 'centro', round(clat, 2), round(clon, 2), 'intensidad', cells[0][2] / 10, 'mm/h')
    assert abs(clat - mza[0]) < 0.15 and abs(clon - mza[1]) < 0.15
    assert max(v for _, _, v in cells) == 60
    # sin lluvia -> sin celdas
    rr[:] = 0.0
    assert recortar(x, y, rr, proy)[0] == []
    print('OK')

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--test': prueba()
    else: main(sys.argv[1] if len(sys.argv) > 1 else 'lluvia.json')
