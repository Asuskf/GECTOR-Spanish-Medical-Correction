# 🏥 MLOps NLP Clínico: Auditoría Hospital Y 

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-%2334D058.svg?logo=huggingface&logoColor=white)](https://huggingface.co/)
[![Polars](https://img.shields.io/badge/Polars-%23CD792C.svg?logo=polars&logoColor=white)](https://pola.rs/)

Este repositorio contiene un pipeline completo de **MLOps y NLP** diseñado para resolver un problema crítico en la gestión de datos hospitalarios: la corrección ortográfica y normalización de historias clínicas redactadas con premura, habilitando el análisis automatizado para sistemas de Business Intelligence (BI).

---

## 🎯 El Problema y el Objetivo

Durante la extracción de datos en el **Hospital Y**, se detectó una alta incidencia de faltas de ortografía y uso de jerga médica no estandarizada (ej. *"cirujia"*, *"urjencia"*). Esto bloqueaba la capacidad de los directivos para agrupar procedimientos y analizar macrotendencias operativas.

**Objetivo Principal:** Implementar un sistema automatizado de corrección gramatical (Sequence-to-Sequence) eficiente y de bajo costo computacional, que actúe como habilitador para la posterior clasificación de textos médicos.

---

## 🧩 Arquitectura del Pipeline

El proyecto está diseñado bajo una metodología rigurosa que pone al **DataOps como piedra angular del MLOps y LLMOps**, dividiendo el problema en fases abordables:

1. **🧹 DataOps (Ingesta y Limpieza):** - Procesamiento estructurado utilizando `Pandas` y `Polars` para asegurar la calidad del texto (manejo de nulos, eliminación de caracteres especiales).
   - Divisiones estratificadas y manejo de desbalanceo de clases para preparar el terreno.
2. **🧠 NLP Core (Corrección Ortográfica):**
   - Implementación de un enfoque de **Low-Cost ML Fine-Tuning**.
   - Uso de la librería personalizada [GECToR Improved Training](https://github.com/Asuskf/gector-improved-training) adaptada para modelos *Tiny*.
   - Entrenamiento basado en `mrm8488/spanish-TinyBERT-betito` para garantizar inferencias ultrarrápidas y bajo consumo de memoria.
3. **🚀 MLOps (Operaciones y Despliegue):**
   - Integración con el ecosistema de Hugging Face (`huggingface_hub`) para el versionado y almacenamiento del artefacto del modelo.
   - Diseño orientado a CI/CD para integraciones futuras en AWS (SageMaker) o GCP (Cloud Functions).
4. **📊 Monitoreo Post-Despliegue (Diseño):**
   - Estrategia planificada para la detección de *Data Drift* y *Concept Drift* utilizando herramientas como Evidently AI o WhyLabs, asegurando el cumplimiento continuo normativo (HIPAA).

---

## 🛠️ Tecnologías Utilizadas

| Categoría | Herramientas |
| :--- | :--- |
| **Manipulación de Datos** | `Polars`, `Pandas`, `NumPy` |
| **Machine Learning & NLP** | `PyTorch`, `Transformers` (Hugging Face), `Scikit-Learn` |
| **Entrenamiento GEC** | `gector` (Custom Fork) |
| **Aceleración y Hardware** | `Accelerate`, GPU (CUDA) |

---

## ⚙️ Uso y Reproducción

### 1. Clonar el repositorio base
Para el entrenamiento, utilizamos una versión optimizada de GECToR:
```bash
git clone [https://github.com/Asuskf/gector.git](https://github.com/Asuskf/gector.git)
cd gector
pip install git+[https://github.com/Asuskf/gector.git](https://github.com/Asuskf/gector.git)
