# NII-UIT at VBS2025: Multimodal Video Retrieval with LLM Integration and Dynamic Temporal Search

> Canonical: Gia, B.T. et al. (2025). *MMM 2025*, LNCS 15524, pp. 318–325.  
> DOI: [10.1007/978-981-96-2074-6_38](https://doi.org/10.1007/978-981-96-2074-6_38)  
> Body below transcribed from team-supplied text (Studocu/PDF fragments). Light cleanup for Markdown; wording kept close to the paper.

---

## 2 Proposed System

The NII-UIT retrieval system (Fig. 1) uses a **two-phase** pipeline: offline and online.

**Offline.** Keyframes are extracted from each shot. Feature vectors of each keyframe are stored in a vector database (**Milvus**). Keyframes are also run through an object detection model; detected objects go into a **tabular** database.

**Online.** Users submit visual or textual queries. Textual queries can be paraphrased with models like **GPT-4o** or used as prompts for generative models to produce images. For Q/A, the user’s question can be transformed into a textual description. Results from different query types are aggregated and filtered by object-query constraints for the final output.

### 2.1 Pre-processing

Processing every frame is inefficient on VBS-scale data. The authors use an optimized keyframe selection inspired by **Vibro**, enhanced with **BEiT-3** for semantic features: extract features from every tenth frame; keep frames with significant differences. Keyframes are saved as **WebP**.

### 2.2 Embedding-based Retrieval

Cross-modal VLMs embed images and text in a shared space. Prior VBS winner VISIONE 5.0 used OpenCLIP ViT-L/14, CLIP2Video, ALADIN; the field also uses BEiT-3, OpenCLIP H-14, InternVL-G, etc.

NII-UIT runs advanced VLMs for textual and visual search. Per-model ranks are **normalized**, then combined with a **data fusion** algorithm to balance strengths.

### 2.3 Query Expansion

Short/vague queries hurt retrieval. **GPT-4o** (or similar LLM) generates **five paraphrases**. Users can pick one, or run paraphrases in parallel and view per-version results or a **fused** ranked list.

### 2.4 Visual Query Generation

**Stable Diffusion** turns text into image queries (inspired by Ma et al.). Users can search with one generated image or fuse results from multiple generated images.

### 2.5 Object Filtering

Embedding search struggles with counting/distinction under overlap, occlusion, scale. **Co-DETR** detects COCO objects on each keyframe; object constraints filter keyframes that lack the required objects.

### 2.6 Multi-modal Search and Dynamic Temporal Search

**Single-stage multi-modal search.** Combine textual, visual, query-expansion, and image-generation channels. Per-shot scores from each channel are **normalized**, then aggregated by **mean pooling**. **Object filtering** then drops shots missing required objects; remaining shots are reranked to the UI.

**Temporal search.** Unlimited chained single stages so queries can be refined iteratively. For **KIS-T**, when it is unclear whether extra text is before or after the first description, they use an enhanced temporal mechanism inspired by **vitrivr**: instead of only testing “before vs after,” they explore **shots surrounding** the initial result, score those neighbors against the new query text, and **rerank by aggregating scores across stages**.

### 2.7 Question Answering (Q/A)

Q/A questions also act as retrieval cues. Example: *“There is also a red Volkswagen Golf GTI at this show. What is the license plate number?”* → textual description *“A red Volkswagen Golf GTI with a license plate”* (manual or LLM), then retrieval via the §2.6 temporal approach (KIS-T style).

### 2.8 User Interface

Web UI with two result modes: (1) group nearby shots for comparison; (2) highlight the top-scoring frame per shot. **Advanced Mode** (Fig. 2): model weights, paraphrasing, SD visual queries. Advanced Mode off for novices.

---

## 3 Conclusion

System uses LLMs for query expansion, multimodal inputs (text, visual, object filter, SD-generated visuals), and dynamic temporal search for a richer evaluation of frame relevance than traditional before/after-only methods.
