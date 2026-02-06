# Guía de Reinicio de Servidor - Nuvant VA

Si el servidor se cae o no responde, sigue estos pasos para levantarlo manualmente en la terminal.

## Método 1: Comando Rápido
Desde la carpeta raíz del proyecto (`Nuvant_VA`), ejecuta:

```bash
fuser -k 8000/tcp || true && ./venv/bin/uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Método 2: Paso a paso
Si prefieres hacerlo manualmente para ver los errores:

1. **Entrar a la carpeta**:
   ```bash
   cd /home/juan-david-valencia/Escritorio/Nuvant_VA
   ```
2. **Activar el entorno**:
   ```bash
   source venv/bin/activate
   ```
3. **Lanzar el servidor**:
   ```bash
   uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload
   ```

## Verificación
Una vez ejecutado, el servidor debería estar disponible en:
`http://localhost:8000/static/index.html`
