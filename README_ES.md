# Multi-Object Tracking & Traffic Counting

Idioma: [🇪🇸 Español](README_ES.md) | [🇬🇧 English](README.md)

Sistema de visión por computador para **detectar, seguir y contar peatones, coches y bicicletas** en un vídeo urbano grabado con cámara fija. El proyecto combina **YOLO11s** con cuatro configuraciones de seguimiento y compara tanto el resultado del conteo como el coste computacional de cada una.

El objetivo no es únicamente obtener un contador: también es estudiar cómo influyen las pérdidas de detección, las oclusiones y los cambios de identidad (*ID switches*) en una aplicación de seguimiento multiobjeto.

## 🎬 Demo



https://github.com/user-attachments/assets/eea570a2-2624-4835-9389-3804011eca77




Comparison of ByteTrack, BoT-SORT, BoT-SORT with Re-ID,
and OC-SORT on the same traffic video

## Funcionalidades

- Detección de personas, bicicletas y coches mediante un modelo YOLO preentrenado.
- Asignación de identificadores y seguimiento temporal con ByteTrack, BoT-SORT, BoT-SORT con Re-ID y OC-SORT.
- Conteo independiente por clase y por sentido de cruce.
- Dos líneas virtuales: una para peatones y bicicletas, y otra para coches.
- Exportación de vídeos anotados, eventos de cruce en CSV y métricas de rendimiento en JSON.
- Comparación reproducible utilizando el mismo vídeo, detector y parámetros de inferencia.

## Tecnologías y arquitectura

**Python · Ultralytics/YOLO11s · OpenCV · PyTorch · NumPy**

```text
Vídeo → Fotogramas → YOLO11s → Tracker → Trayectorias con ID
                                        ↓
                                 Lógica de conteo
                                        ↓
                    Vídeo anotado + events.csv + summary.json
```

YOLO localiza y clasifica los objetos de cada fotograma. El tracker asocia las detecciones entre fotogramas para mantener sus IDs. Finalmente, una lógica de conteo independiente transforma los cambios de lado de las trayectorias en eventos.

### Configuración común de los experimentos

| Parámetro | Valor |
|---|---|
| Detector | `yolo11s.pt`, preentrenado en COCO |
| Clases COCO | Persona (`0`), bicicleta (`1`) y coche (`2`) |
| Tamaño de entrada (`imgsz`) | `960` |
| Confianza mínima (`conf`) | `0.10` |
| Vídeo de entrada | Cámara fija, 30 FPS, aproximadamente 36 s |
| Dispositivo utilizado para las mediciones | CPU |
| Fotogramas incluidos en las métricas temporales | 1.072 (se excluyen los 10 primeros) |

La línea destinada a peatones y bicicletas utiliza como punto de referencia el **centro inferior** de la caja detectada; la línea de coches utiliza el **centro** de su caja. Se utilizan bandas de tolerancia, confirmación del lado e intervalos mínimos entre eventos para reducir conteos repetidos. Las bicicletas disponen de un contador independiente, aunque comparten la línea de los peatones.

El sistema cuenta **eventos de cruce**, no personas únicas a lo largo de distintas sesiones. Los IDs asignados por un tracker son locales y pueden cambiar cuando se pierde una trayectoria.

## Estructura del proyecto

```text
Traffic_Tracking/
├── main.py                 # Detección, tracking, conteo y métricas
├── compare.py              # Composición visual de los cuatro vídeos
├── run_all.ps1             # Ejecución secuencial de experimentos
├── bot_reid.yaml           # Configuración de BoT-SORT con Re-ID
├── videos/
│   └── video.mp4           # Vídeo de prueba (archivo local)
├── outputs/
│   ├── tracking_<tracker>.mp4
│   ├── events_<tracker>.csv
│   └── summary_<tracker>.json
└── README.md
```

Los archivos `compare.py` y `run_all.ps1` permiten, respectivamente, visualizar las ejecuciones en paralelo y automatizar los experimentos, cuando están incluidos en el repositorio. Los archivos originales de vídeo deben publicarse únicamente si su licencia lo permite.

## Instalación y ejecución

En Windows, desde el terminal de VS Code o PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install ultralytics opencv-python numpy torch
```

Coloca el vídeo en `videos/video.mp4`. La instalación debe incluir las configuraciones de los trackers utilizadas durante los experimentos; `bot_reid.yaml` debe tener activado `with_reid: True`.

Ejecuta un algoritmo con:

```powershell
python main.py --tracker bytetrack
python main.py --tracker botsort
python main.py --tracker botsort_reid
python main.py --tracker ocsort
```

Si tienes el script de automatización, puedes ejecutar los cuatro consecutivamente mediante `.\run_all.ps1`. La compatibilidad y disponibilidad de `ocsort.yaml` dependen de la instalación de tracking utilizada.

Cada ejecución guarda un vídeo anotado, un CSV con `frame`, `time_s`, `class`, `track_id` y `direction`, y un JSON con los contadores y los tiempos medidos.

## Resultados de conteo

La referencia se obtuvo mediante un conteo manual del vídeo: **21 peatones, 45 coches y 1 bicicleta** (67 cruces en total).

| Algoritmo | Peatones | Coches | Bicicletas | Total | Error absoluto por categoría |
|---|---:|---:|---:|---:|---:|
| **Referencia manual** | **21** | **45** | **1** | **67** | — |
| ByteTrack | 22 | 45 | 0 | 67 | 2 |
| BoT-SORT | 21 | 45 | 1 | 67 | 0 |
| BoT-SORT + Re-ID | 21 | 45 | 1 | 67 | 0 |
| OC-SORT | 22 | 45 | 0 | 67 | 2 |

El error absoluto por categoría es la suma de las diferencias absolutas entre el conteo previsto y el real para las tres clases.

**Interpretación:**

- Los cuatro algoritmos registran **45 coches**, coincidiendo con la referencia manual.
- BoT-SORT, con y sin Re-ID, reproduce el conteo total de cada clase en este vídeo.
- ByteTrack y OC-SORT obtienen también **67 eventos**, pero registran un peatón adicional y no contabilizan la bicicleta. El total correcto oculta, por tanto, dos errores por categoría.
- En la inspección del vídeo, el caso de la bicicleta está relacionado con la detección simultánea del ciclista como persona y con la asociación y las condiciones del evento de cruce. No se ha aislado todavía una causa única: podría intervenir la continuidad del ID, la geometría de la línea o la lógica del contador.

La coincidencia de los totales **no demuestra ausencia de falsos positivos o cruces omitidos**. Para calcular precisión y *recall* de eventos sería necesario emparejar cada cruce registrado con una anotación manual que incluya su instante y dirección.

## Resultados de rendimiento

Las mediciones se realizaron en **CPU**, con el mismo detector, vídeo y resolución. Se excluyeron los 10 primeros fotogramas para reducir el efecto de la inicialización.

| Algoritmo | Detección + tracking (ms/frame) | Ciclo completo (ms/frame) | P95 del ciclo (ms) | FPS efectivos |
|---|---:|---:|---:|---:|
| **ByteTrack** | **162,53** | **184,23** | **194,30** | **5,43** |
| BoT-SORT | 191,18 | 212,85 | 224,56 | 4,70 |
| BoT-SORT + Re-ID | 205,78 | 227,62 | 239,17 | 4,39 |
| OC-SORT | 165,36 | 187,29 | 197,77 | 5,34 |

El tiempo de **detección + tracking** corresponde a la llamada completa a YOLO y al tracker: no es el coste aislado del algoritmo de asociación. El **ciclo completo** incluye, además, la lectura de fotogramas, el conteo, la anotación y la escritura del vídeo. El P95 indica el tiempo por debajo del cual queda aproximadamente el 95 % de los ciclos medidos.

**Interpretación:**

- **ByteTrack** es la configuración más rápida, con 5,43 FPS efectivos; **OC-SORT** ofrece un rendimiento cercano, con 5,34 FPS.
- **BoT-SORT** alcanza 4,70 FPS, a cambio de un mayor tiempo de procesamiento en esta configuración.
- **Activar Re-ID** aumenta el tiempo medio del ciclo de BoT-SORT de 212,85 a 227,62 ms (aproximadamente **6,9 %**) y reduce los FPS efectivos de 4,70 a 4,39. En este vídeo no cambia los totales del conteo.
- Ninguna ejecución alcanza los 30 FPS del vídeo original en la CPU utilizada. Son pruebas de procesamiento *offline*, no una demostración de funcionamiento en tiempo real.

Estas cifras corresponden a una ejecución por configuración. Para establecer diferencias de rendimiento más sólidas habría que repetir las pruebas, controlar la carga del equipo y registrar las versiones exactas de las dependencias.

## Análisis del seguimiento y limitaciones

La revisión visual mostró dificultades para mantener los IDs cuando varios peatones caminan muy juntos o se cruzan. La configuración con Re-ID se incluyó para comprobar si la apariencia visual ayuda a resolver esas asociaciones ambiguas; sin embargo, los datos actuales **no cuantifican los ID switches ni las fragmentaciones de trayectorias**.

Los CSV exportados contienen únicamente **eventos de cruce**, no todas las posiciones de cada ID en cada fotograma. Por ese motivo, no es posible calcular con ellos métricas como IDF1, HOTA o MOTA ni afirmar que un tracker conserva mejor las identidades basándose únicamente en el conteo final.

Otras limitaciones del prototipo son la evaluación sobre un único vídeo, la cámara fija, la posible doble interpretación de un ciclista como persona y bicicleta, y la sensibilidad de los resultados a las líneas y a los parámetros de confirmación. El Re-ID empleado tampoco implica reidentificación global entre cámaras o sesiones.

## Próximas mejoras

1. **Distinguir peatones y ciclistas:** asociar las detecciones de persona y bicicleta cuando correspondan a un ciclista, sin confundirlo con una persona que camina junto a una bicicleta.
2. **Perfeccionar el cruce de línea:** calcular la intersección de la trayectoria con el segmento y revisar los umbrales de confirmación para objetos rápidos.
3. **Evaluar el tracking de forma objetiva:** exportar trayectorias por fotograma, anotar casos de oclusión y contabilizar cambios y recuperaciones de ID.
4. **Ampliar la evaluación:** probar otros vídeos, repetir las mediciones de rendimiento y estudiar el compromiso entre precisión, resolución y velocidad.

## Conclusiones

En este escenario, **BoT-SORT ofrece el mejor resultado de conteo por categorías**, con 21 peatones, 45 coches y 1 bicicleta. **ByteTrack es la opción más rápida** y OC-SORT presenta un rendimiento muy similar. La activación de Re-ID añade coste computacional sin modificar el conteo final; su posible ventaja en la continuidad de los IDs requiere una evaluación específica del seguimiento.

El principal aprendizaje es que un total aparentemente perfecto no garantiza un sistema correcto: resulta imprescindible separar la calidad de la detección, la continuidad del tracking y las reglas de conteo para identificar dónde se produce cada error.
