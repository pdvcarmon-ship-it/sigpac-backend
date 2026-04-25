from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import httpx
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.mask import mask as rasterio_mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import geopandas as gpd
from shapely.geometry import shape, box
import json
import io
import os
import hashlib
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SIGPAC Sentinel API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

COPERNICUS_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
COPERNICUS_SEARCH_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
COPERNICUS_DOWNLOAD_URL = "https://download.dataspace.copernicus.eu/odata/v1/Products"

SIGPAC_WFS_BASE = "https://www.fega.gob.es/PwfgbWfsWeb/rest/wfs"

COPERNICUS_USER = os.getenv("COPERNICUS_USER", "")
COPERNICUS_PASS = os.getenv("COPERNICUS_PASS", "")

_token_cache = {"token": None, "expires_at": 0}


async def get_copernicus_token() -> str:
    now = time.time()
    if _token_cache["token"] and now < _token_cache["expires_at"] - 60:
        return _token_cache["token"]

    if not COPERNICUS_USER or not COPERNICUS_PASS:
        raise HTTPException(
            status_code=500,
            detail="Copernicus credentials not configured. Set COPERNICUS_USER and COPERNICUS_PASS env vars."
        )

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            COPERNICUS_TOKEN_URL,
            data={
                "grant_type": "password",
                "username": COPERNICUS_USER,
                "password": COPERNICUS_PASS,
                "client_id": "cdse-public",
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        _token_cache["token"] = data["access_token"]
        _token_cache["expires_at"] = now + data.get("expires_in", 3600)
        return _token_cache["token"]


def cache_key(prefix: str, **kwargs) -> str:
    key = json.dumps(kwargs, sort_keys=True)
    return hashlib.md5(f"{prefix}_{key}".encode()).hexdigest()


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.get("/sigpac/parcela")
async def get_parcela(
    provincia: int = Query(..., description="Código provincia (ej: 28)"),
    municipio: int = Query(..., description="Código municipio"),
    poligono: int = Query(..., description="Número polígono"),
    parcela: int = Query(..., description="Número parcela"),
):
    """Obtiene la geometría de una parcela SIGPAC vía WFS."""
    ck = cache_key("sigpac", prov=provincia, mun=municipio, pol=poligono, par=parcela)
    cache_file = CACHE_DIR / f"sigpac_{ck}.geojson"

    if cache_file.exists():
        logger.info(f"Cache hit SIGPAC: {ck}")
        return JSONResponse(content=json.loads(cache_file.read_text()))

    ref = f"{provincia:02d}{municipio:03d}0{poligono:04d}{parcela:05d}0000"

    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeNames": "sigpac:parcela",
        "outputFormat": "application/json",
        "CQL_FILTER": f"referencia_sigpac='{ref}'",
        "count": "1",
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(SIGPAC_WFS_BASE, params=params)
            resp.raise_for_status()
            data = resp.json()

        if not data.get("features"):
            raise HTTPException(status_code=404, detail=f"Parcela no encontrada: {ref}")

        cache_file.write_text(json.dumps(data))
        return JSONResponse(content=data)

    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Error WFS SIGPAC: {str(e)}")


@app.get("/sentinel/buscar")
async def buscar_imagenes(
    bbox: str = Query(..., description="min_lon,min_lat,max_lon,max_lat"),
    fecha_inicio: str = Query(..., description="YYYY-MM-DD"),
    fecha_fin: str = Query(..., description="YYYY-MM-DD"),
    max_nubosidad: float = Query(20.0, description="% máximo nubes"),
):
    """Busca imágenes Sentinel-2 disponibles para una zona y fecha."""
    try:
        min_lon, min_lat, max_lon, max_lat = map(float, bbox.split(","))
    except ValueError:
        raise HTTPException(status_code=400, detail="bbox inválido. Formato: min_lon,min_lat,max_lon,max_lat")

    footprint = f"POLYGON(({min_lon} {min_lat},{max_lon} {min_lat},{max_lon} {max_lat},{min_lon} {max_lat},{min_lon} {min_lat}))"

    params = {
        "$filter": (
            f"Collection/Name eq 'SENTINEL-2' "
            f"and ContentDate/Start gt {fecha_inicio}T00:00:00.000Z "
            f"and ContentDate/Start lt {fecha_fin}T23:59:59.000Z "
            f"and Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' "
            f"and att/OData.CSC.DoubleAttribute/Value le {max_nubosidad}) "
            f"and OData.CSC.Intersects(area=geography'SRID=4326;{footprint}')"
        ),
        "$orderby": "ContentDate/Start desc",
        "$top": "10",
        "$expand": "Attributes",
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(COPERNICUS_SEARCH_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

        productos = []
        for item in data.get("value", []):
            cloud = next(
                (a["Value"] for a in item.get("Attributes", []) if a["Name"] == "cloudCover"),
                None,
            )
            productos.append({
                "id": item["Id"],
                "nombre": item["Name"],
                "fecha": item["ContentDate"]["Start"][:10],
                "nubosidad": round(cloud, 1) if cloud is not None else None,
                "size_mb": round(item.get("ContentLength", 0) / 1e6, 1),
            })

        return {"total": len(productos), "productos": productos}

    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Error búsqueda Copernicus: {str(e)}")


INDICES = {
    "NDVI": {
        "formula": "(B08 - B04) / (B08 + B04 + 1e-10)",
        "descripcion": "Normalized Difference Vegetation Index",
        "cmap": "RdYlGn",
        "vmin": -1,
        "vmax": 1,
        "bandas": ["B04", "B08"],
    },
    "NDWI": {
        "formula": "(B03 - B08) / (B03 + B08 + 1e-10)",
        "descripcion": "Normalized Difference Water Index",
        "cmap": "Blues",
        "vmin": -1,
        "vmax": 1,
        "bandas": ["B03", "B08"],
    },
    "EVI": {
        "formula": "2.5 * (B08 - B04) / (B08 + 6*B04 - 7.5*B02 + 1 + 1e-10)",
        "descripcion": "Enhanced Vegetation Index",
        "cmap": "YlGn",
        "vmin": -1,
        "vmax": 1,
        "bandas": ["B02", "B04", "B08"],
    },
    "NDRE": {
        "formula": "(B08 - B05) / (B08 + B05 + 1e-10)",
        "descripcion": "Normalized Difference Red Edge",
        "cmap": "RdYlGn",
        "vmin": -1,
        "vmax": 1,
        "bandas": ["B05", "B08"],
    },
    "SAVI": {
        "formula": "1.5 * (B08 - B04) / (B08 + B04 + 0.5 + 1e-10)",
        "descripcion": "Soil-Adjusted Vegetation Index",
        "cmap": "YlGn",
        "vmin": -1,
        "vmax": 1,
        "bandas": ["B04", "B08"],
    },
}


@app.get("/indices/lista")
async def lista_indices():
    """Devuelve los índices disponibles."""
    return {
        k: {"descripcion": v["descripcion"], "bandas": v["bandas"]}
        for k, v in INDICES.items()
    }


@app.get("/indice/calcular")
async def calcular_indice(
    producto_id: str = Query(..., description="ID producto Sentinel"),
    indice: str = Query(..., description="NDVI|NDWI|EVI|NDRE|SAVI"),
    bbox: Optional[str] = Query(None, description="Recorte: min_lon,min_lat,max_lon,max_lat"),
    formato: str = Query("png", description="png|geotiff|stats"),
):
    """Descarga bandas Sentinel, calcula índice y devuelve imagen."""
    indice = indice.upper()
    if indice not in INDICES:
        raise HTTPException(status_code=400, detail=f"Índice desconocido. Disponibles: {list(INDICES.keys())}")

    cfg = INDICES[indice]

    ck = cache_key("indice", pid=producto_id, idx=indice, bbox=bbox)
    cache_png = CACHE_DIR / f"{ck}.png"
    cache_stats = CACHE_DIR / f"{ck}_stats.json"

    if cache_png.exists() and formato == "png":
        logger.info(f"Cache hit índice PNG: {ck}")
        return StreamingResponse(io.BytesIO(cache_png.read_bytes()), media_type="image/png")

    if cache_stats.exists() and formato == "stats":
        return JSONResponse(content=json.loads(cache_stats.read_text()))

    try:
        token = await get_copernicus_token()
    except HTTPException:
        return _demo_indice(indice, cfg, bbox, cache_png, cache_stats, formato)

    bandas_data = {}
    async with httpx.AsyncClient(
        headers={"Authorization": f"Bearer {token}"},
        timeout=120,
        follow_redirects=True,
    ) as client:
        for banda in cfg["bandas"]:
            url = f"{COPERNICUS_DOWNLOAD_URL}({producto_id})/Nodes({producto_id}.SAFE)/Nodes(GRANULE)/Nodes/Nodes(IMG_DATA)/Nodes(R10m)/Nodes({banda}.jp2)/$value"
            logger.info(f"Descargando {banda}...")
            resp = await client.get(url)
            if resp.status_code != 200:
                raise HTTPException(status_code=502, detail=f"No se pudo descargar {banda}: {resp.status_code}")
            bandas_data[banda] = resp.content

    arrays = {}
    transform_ref = None
    crs_ref = None
    for banda, raw in bandas_data.items():
        with MemoryFile(raw) as mf:
            with mf.open() as src:
                if bbox:
                    min_lon, min_lat, max_lon, max_lat = map(float, bbox.split(","))
                    geom = [box(min_lon, min_lat, max_lon, max_lat).__geo_interface__]
                    arr, transform = rasterio_mask(src, geom, crop=True)
                else:
                    arr = src.read(1).astype(np.float32)
                    transform = src.transform
                arrays[banda] = arr.astype(np.float32) / 10000.0
                if transform_ref is None:
                    transform_ref = transform
                    crs_ref = src.crs

    formula = cfg["formula"]
    for banda, arr in arrays.items():
        formula = formula.replace(banda, f"arrays['{banda}']")
    resultado = eval(formula)

    resultado = np.clip(resultado, cfg["vmin"], cfg["vmax"])

    stats = {
        "indice": indice,
        "min": float(np.nanmin(resultado)),
        "max": float(np.nanmax(resultado)),
        "mean": float(np.nanmean(resultado)),
        "std": float(np.nanstd(resultado)),
        "producto_id": producto_id,
    }
    cache_stats.write_text(json.dumps(stats))

    if formato == "stats":
        return JSONResponse(content=stats)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    im = ax.imshow(resultado, cmap=cfg["cmap"], vmin=cfg["vmin"], vmax=cfg["vmax"])
    plt.colorbar(im, ax=ax, label=indice, fraction=0.046, pad=0.04)
    ax.set_title(f"{indice} - {cfg['descripcion']}", fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close()
    buf.seek(0)
    png_bytes = buf.read()
    cache_png.write_bytes(png_bytes)

    return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")


def _demo_indice(indice, cfg, bbox, cache_png, cache_stats, formato):
    """Genera imagen demo cuando no hay credenciales Copernicus."""
    np.random.seed(42)
    size = (256, 256)
    resultado = np.random.uniform(cfg["vmin"] * 0.3, cfg["vmax"] * 0.9, size).astype(np.float32)
    resultado = np.clip(resultado, cfg["vmin"], cfg["vmax"])

    stats = {
        "indice": indice,
        "min": float(np.nanmin(resultado)),
        "max": float(np.nanmax(resultado)),
        "mean": float(np.nanmean(resultado)),
        "std": float(np.nanstd(resultado)),
        "modo": "DEMO - configura COPERNICUS_USER y COPERNICUS_PASS",
    }
    cache_stats.write_text(json.dumps(stats))

    if formato == "stats":
        return JSONResponse(content=stats)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    im = ax.imshow(resultado, cmap=cfg["cmap"], vmin=cfg["vmin"], vmax=cfg["vmax"])
    plt.colorbar(im, ax=ax, label=indice, fraction=0.046, pad=0.04)
    ax.set_title(f"{indice} - DEMO (sin credenciales)", fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close()
    buf.seek(0)
    png_bytes = buf.read()
    cache_png.write_bytes(png_bytes)

    return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")


@app.get("/cache/info")
async def cache_info():
    """Info sobre el caché actual."""
    files = list(CACHE_DIR.glob("*"))
    total_mb = sum(f.stat().st_size for f in files if f.is_file()) / 1e6
    return {
        "archivos": len(files),
        "total_mb": round(total_mb, 2),
        "directorio": str(CACHE_DIR.absolute()),
    }


@app.delete("/cache/limpiar")
async def limpiar_cache(dias: int = Query(7, description="Eliminar archivos más antiguos que N días")):
    """Limpia archivos de caché antiguos."""
    cutoff = time.time() - dias * 86400
    eliminados = 0
    for f in CACHE_DIR.glob("*"):
        if f.is_file() and f.stat().st_mtime < cutoff:
            f.unlink()
            eliminados += 1
    return {"eliminados": eliminados}
