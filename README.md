# JS-PotatoRAG 🥔 - Local AI Vector Console & PWA

**JS-PotatoRAG** es una consola RAG (Retrieval-Augmented Generation) de inteligencia artificial local, rápida, eficiente y ejecutable en un solo proceso. Es una **Aplicación Web Progresiva (PWA)** diseñada para funcionar con cero dependencias externas o de internet (totalmente air-gapped).

Combina una base de datos vectorial ultraligera escrita en Rust y compilada a **WebAssembly (WASM)**, extracción de embeddings locales en **ONNX Runtime** y un LLM local en proceso (**Gemma 3 270M IT**) o integraciones en caliente con daemons locales como **Ollama** y **LM Studio** (OpenAI compatible).

---

## 🚀 Características Principales

*   **PWA Instalable e Instantánea**: Convierte la interfaz en una aplicación de escritorio o móvil con precarga estática offline mediante un Service Worker.
*   **Búsqueda Vectorial WASM (Rust)**: Base de datos vectorial indexada en 3 bits con polarización angular (`PolarQuantizedStore`), ofreciendo un **7.1x de velocidad** comparado con JavaScript puro (escaneando 10,000 vectores en solo **14.9 ms**).
*   **Ahorro de Memoria de 21.3x**: Compresión vectorial extrema que permite meter **1 millón de vectores de 768-D en solo 137 MB de RAM**.
*   **Embeddings ONNX Locales**: Extracción integrada mediante `@huggingface/transformers` con el modelo `embeddinggemma-300m` quantizado en 8 bits (`q8`), procesado en **~158 ms** en CPU.
*   **Generador LLM In-Process**: Soporta el nuevo modelo **Google Gemma 3 270M (IT - q4)** de 150 MB, ejecutándose de forma local dentro del proceso de Node.js a más de **27 tokens/segundo** (sin necesidad de Ollama o LM Studio activos).
*   **Soporte Multi-Proveedor**: Panel de ajustes avanzado en la barra lateral para alternar en caliente entre el generador ONNX interno, Ollama y servidores OpenAI compatibles (LM Studio, vLLM, LocalAI).
*   **Aislamiento y Autoescalado Vectorial**: Escala dinámicamente las dimensiones del vector en la base de datos WASM y aísla los archivos por modelo (`docs_<model>_<dim>.p3.bin`) para evitar colisiones.

---

## 📐 Arquitectura del Sistema

El siguiente diagrama muestra el flujo de datos autocontenido de JS-PotatoRAG:

```mermaid
graph TD
    User([Usuario]) -->|Pregunta / Ingesta| UI[Interfaz PWA - Cliente]
    UI -->|Service Worker - Offline Cache| SW[sw.js]
    UI -->|API POST Request /api/*| Server[Servidor Express - server.js]
    
    subgraph Servidor local de Node.js
        Server -->|Embeddings| EmbSource{Origen Embeddings}
        EmbSource -->|Local ONNX| ONNX_Emb[embeddinggemma-300m q8]
        EmbSource -->|Ollama API| Ollama_Emb[nomic-embed-text / etc]
        EmbSource -->|LM Studio API| LM_Emb[OpenAI v1/embeddings]
        
        ONNX_Emb -->|Vector de embeddings| VDB_Factory[getStore dim]
        Ollama_Emb -->|Vector de embeddings| VDB_Factory
        LM_Emb -->|Vector de embeddings| VDB_Factory
        
        VDB_Factory -->|WASM FFI| WASM[rust_polar.wasm]
        WASM -->|Cuantización 3-bit| BinDB[(docs_model_dim.p3.bin)]
        
        Server -->|Chat LLM| LLM_Source{Proveedor de LLM}
        LLM_Source -->|Local ONNX| ONNX_LLM[Gemma-3 270M IT q4]
        LLM_Source -->|Ollama Native| Ollama_LLM[Ollama Serve api/chat]
        LLM_Source -->|LM Studio / OpenAI| LM_LLM[LM Studio / OpenAI v1/chat]
    end
    
    ONNX_LLM -->|Stream con TextStreamer| Server
    Ollama_LLM -->|Stream JSON-lines| Server
    LM_LLM -->|Stream SSE OpenAI| Server
    Server -->|Event Stream| UI
```

---

## ⚡ Pruebas de Rendimiento (Benchmark Local)

Pruebas ejecutadas localmente sobre una base de datos de **10,000 vectores sintéticos** de **768 dimensiones**:

*   **Ingesta y Cuantización (WASM Rust)**: **160 ms** para procesar e indexar 10k vectores (1.3x más rápido que JS puro).
*   **Búsqueda Vectorial WASM (top-10)**: **14.98 ms** avg/consulta (frente a 105.68 ms en JS puro, **7.1x de velocidad**).
*   **Extracción de Embeddings Local ONNX**: **158.84 ms** por fragmento de texto.
*   **Inferencia LLM Gemma 3 270M local**: **27.41 tokens/segundo** (carga del modelo en memoria en solo **1.24s**).

---

## 🛠️ Instalación y Uso

### 1. Requisitos Previos
*   [Node.js](https://nodejs.org/) (Versión 18 o superior recomendada, desarrollado en Node v24).
*   (Opcional) Ollama o LM Studio ejecutándose localmente si deseas usar modelos externos más grandes.

### 2. Clonar e Instalar Dependencias
```bash
git clone https://github.com/tu-usuario/JS-PotatoRAG.git
cd JS-PotatoRAG
npm install
```

### 3. Iniciar la Aplicación
```bash
npm start
```
El servidor backend se iniciará en **`http://localhost:3005`**.

### 4. Probar en el Navegador
Abre `http://localhost:3005` en tu navegador:
*   Para un funcionamiento **100% desconectado e in-process**: Selecciona **LLM Provider: Local ONNX (Gemma-3 270M)** y **Embedding Source: Local ONNX (embeddinggemma)** en la configuración lateral. El primer mensaje descargará y cacheará automáticamente los modelos (~150MB y ~300MB) y funcionará de forma air-gapped.
*   Para usar **LM Studio**: Enciende el servidor de LM Studio en el puerto `1234`, carga un modelo de chat y de embeddings, y ajusta el panel lateral a `LM Studio / OpenAI compatible` y base URL `http://localhost:1234/v1`.
*   Para usar **Ollama**: Enciende Ollama en el puerto `11434` y selecciona `Ollama Native API`.

---

## ⚙️ Estructura del Proyecto

*   `server.js`: Servidor Express que gestiona el enrutamiento de peticiones, la carga de los modelos ONNX y la decodificación de flujos SSE.
*   `wasm-vector-store.cjs` y `wasm-polar-store.cjs`: Wrappers FFI de JavaScript que gestionan los punteros de memoria lineal para transferir datos al módulo WebAssembly.
*   `rust_polar.wasm`: Binario de WebAssembly compilado a partir del motor Rust.
*   `public/`: Carpeta que contiene la aplicación PWA (HTML, estilos CSS, manifiesto de instalación y Service Worker).
*   `rust_polar/`: Código fuente de Rust del motor de búsqueda y cuantización polar de 3 bits.
*   `benchmark_wasm.cjs`: Script para comparar el rendimiento de búsqueda entre WASM y JavaScript puro.
*   `benchmark_onnx.js`: Script para medir los tiempos de inferencia del extractor de embeddings local.

---

## 📄 Licencia

Este proyecto está bajo la licencia MIT.
