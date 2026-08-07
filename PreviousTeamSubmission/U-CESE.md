# U-CESE

> Source: `U-CESE.pdf`  
> Extracted for easier reading. Formulas may need cleanup if PDF used special fonts.

---

arXiv:2605.23274v1 [cs.CV] 22 May 2026  U-CESE: Unified Clip-based Event Search Engine
                                                      for AI Challenge HCMC 2025

                                            Duc-Nhuan Le∗,†1,2 , Hoang-Phuc Nguyen†1,2 , Thanh-Duy Lam†1,2 ,
                                                            Minh-Nhut Dang†1,2 , and Minh-Hoang Le†1,2

                                                 1 Faculty of Information Technology, University of Science, VNU-HCM
                                                         2 Vietnam National University, Ho Chi Minh City, Vietnam

                                             {ldnhuan22,nhphuc222,ltduy22,dmnhut22,lmhoang22}@apcs.fitus.edu.vn

                                                 Abstract. Retrieving events from large-scale video datasets is challeng-
                                                 ing due to complex temporal, spatial, and multimodal information. This
                                                 paper presents U-CESE, our solution for the AI Challenge HCMC 2025,
                                                 a Unified Clip-based Event Search Engine for multimodal event retrieval
                                                 across diverse video sources. Building on CESE, U-CESE integrates its
                                                 three modules into a single cohesive framework, ensuring consistent pro-
                                                 cessing and retrieval across query types. A core component is the Uni-
                                                 fied Clipping Algorithm, which merges separate clipping algorithms into
                                                 one efficient pipeline. To handle large-scale data, we propose DAKE, a
                                                 lightweight, training-free keyframe extraction method using JPEG file
                                                 size variations to identify significant scene changes. Finally, we introduce
                                                 ReCap, a temporally consistent captioning framework inspired by Re-
                                                 current Neural Network, generating detailed and context-aware textual
                                                 descriptions. Experiments show that U-CESE delivers robust, consistent,
                                                 and efficient performance in large-scale multimodal event retrieval.

                                                 Keywords: Video Event Retrieval, Text-to-Image Matching, Visual Ques-
                                                 tion Answering

                                        1 Introduction

                                        The AI Challenge HCMC (AIC) 2025 [7] tasks participants with developing
                                        systems for retrieving specific events from diverse video sources. Building on
                                        the structure of the Video Browser Showdown (VBS) [17] and the Lifelog Search
                                        Challenge (LSC) [14], AIC 2025 [7] retains the previous format while introducing
                                        a new query type requiring the retrieval of multiple scenes within a video. The
                                        dataset has also been significantly expanded in scale and diversity, demanding
                                        efficient extraction and utilization of spatial, temporal, and object-level features
                                        from large-scale video collections. These capabilities are particularly valuable
                                        for navigating extensive TV news archives, where rapid access to relevant visual
                                        content enhances search and analysis workflows [8].

                                            ∗Corresponding author(s). E-mail(s): ldnhuan22@apcs.fitus.edu.vn
                                            †These authors contributed equally to this research.

2        Nhuan et al.

   User    Text          MobileClip      ……         Unified      TextualDB
         Queries       Vision Encoder  …           Clipping      VisualDB
                                                   Algorithm

          Image         MobileClip          …
         Queries       Text Encoder       …
                                          Queries
                                       Embeddings

                                                   Suggestions

                  Fig. 1. Overall system architecture of U-CESE

    To address these challenges, we present U-CESE, a Unified Clip-based Event
Search Engine for AIC [7]. Our system extends the CESE framework [15],
which retrieves coherent clips matching event descriptions across multiple queries
rather than single frames. However, CESE employs three separate modules, each
with distinct user interfaces and re-ranking strategies, leading to inconsistency
and inefficiency. U-CESE overcomes these limitations by unifying all modalities
into a single cohesive module. As illustrated in Fig. 1, user queries are processed
through our Unified Clipping Algorithm, which replaces CESE’s multiple
clipping procedures with a single streamlined framework. To process large-scale
video data, we further introduce two novel frameworks: DAKE, a training-free
keyframe extraction method based on JPEG file sizes that avoids complex shot
boundary models [22,28]; and ReCap, a temporally consistent captioning ap-
proach inspired by recurrent neural networks (RNN) [11], enabling context-aware
captions for keyframes.

    Our main contributions are summarized as follows:

 – We propose U-CESE, a unified clip-based event search engine that consol-
     idates all modalities into a single retrieval framework.

 – We design the Unified Clipping Algorithm, merging CESE’s three sep-
     arate clipping methods into one consistent and efficient pipeline.

 – We introduce DAKE, a lightweight, training-free keyframe extraction strat-
     egy leveraging JPEG file size variance.

 – We develop ReCap, a temporally consistent captioning model inspired by
     RNNs for detailed and context-preserving descriptions.

2 Related Work

In recent years, video retrieval has evolved into a deep learning task integrat-
ing visual feature extraction, linguistic feature extraction, and embedding-based
retrieval [27]. For visual feature extraction, FFmpeg3 is commonly used for

    3https://ffmpeg.org

U-CESE: Unified Clip-based Event Search Engine  3

keyframe extraction [18], while OpenCLIP [19] serves as a backbone for frame
embedding [18,12,6]. Additional models include TransNetV2 [22] and AutoShot [28]
for shot boundary detection [6], Grounding DINO for object detection [6], and
multimodal models such as InternVL [4], BEiT-3 [25], and BLIP-2 [16] for
enhanced representation [12,6]. For linguistic feature extraction, Whisper [21]
is the primary ASR model used in several systems [18,12,6], while Vintern-
1B [9] is employed for image captioning [18]. PARSeq [2] is fine-tuned for Viet-
namese OCR [12], addressing the limitations of DeepSolo [26]. For embedding-
based retrieval, similarity search is implemented using Elasticsearch [1,18,15,12],
FAISS [10,18], Milvus [24,12,6], and Polars4 [6]. Query reformulation with LLMs
further enhances retrieval precision [12].

    CESE [15] introduces a clip-based event search approach that focuses on
identifying relevant clips rather than individual frames. Queries can contain mul-
tiple scene descriptions, and a clip is considered more relevant if it contains more
unique scenes. CESE uses three clipping algorithms: Visual Semantic, Textual
Semantic, and Keywords Clipping. Each utilizes visual, textual, or keyword cues
to generate candidate clips. While this multi-modal design enhances retrieval
robustness, CESE suffers from inconsistency and inefficiency as it relies on three
separate modules, each with distinct user interfaces and re-ranking strategies,
leading to fragmented workflows and limited system scalability.

3 Challenge Overview

3.1 Dataset Overview

The dataset in the AI Challenge HCMC 2025 - Event Retrieval from Visual
Data consists of 1478 videos from various TV programs with a total duration of
324 hours. For each video in the dataset, the organizer provides its extracting
keyframes and metadata. For each keyframe, objects detected from the Faster
RCNN pretrained OpenImagesV4 model [13] and CLIP features from CLIP ViT-
B/32 model [20] are also provided.

3.2 Query types

The challenge consists of 4 distinct query types:

 – Known-Item Search (KIS): Teams are required to locate any frame cor-
     responding to a specific moment in the dataset within 5 minutes. Five textual
     hints are provided progressively at 1-minute intervals.

 – Video Known-Item Search (VKIS): Instead of textual hints, teams are
     shown a 20-second video segment from the dataset and must retrieve the
     same moment within 4 minutes. Recording the video is not allowed.

 – Visual Question Answering (VQA): Similar to the KIS task, but in-
     cludes an additional question about the identified moment.

    4https://pola.rs/

4  Nhuan et al.

Table 1. The dataset contains 9 TV Programs. The table informs the TV program’s
name, broadcast period, number of videos, and average duration per video. There are
a total of 1478 videos with a total duration of 324 hours.

   TV Program                          Period Videos Mins/video

   "HTV 60 Giay"                       08/24 - 08/25 665  20

   National Bicycle Race HCMC TV Cup 2024 04/24  25       6.5

   Lion Dance - Cho Lon Cup            01/24     43       9

       Thanh Nien Newspaper:           05/24 - 06/24 88   25
   "Bi quyet on thi THPT 2024"

   "Mon ngon moi ngay"                 04/20 - 09/24 498  5

   "Viet Nam di la ghien" Season 3     06/24 - 09/24 16   6

   "Tan man Me Kong"                   05/24 - 09/24 24   19

   "Ðoi mat Me Kong"                   05/24 - 09/24 23   18

             Tuoi Tre Newspaper:       08/24 - 09/24 96   3.5
   "Lan toa nang luong tich cuc 2024"

   Total (324 hours)                             1478     13.4

 – Temporal Alignment (TRAKE): Participants must determine the exact
     frames corresponding to several key moments of a single action and arrange
     them in the correct order.

4 System Overview

This section presents U-CESE, our unified clip-based event search system, il-
lustrated in Fig. 1. The pipeline begins with database construction from raw
video data, as described in Sec. 4.1. This process integrates our two novel frame-
works: DAKE, which performs dynamic-aware keyframe extraction, and Re-
Cap, which generates temporally consistent shot captions. U-CESE supports
both image and text queries. These inputs are embedded using MobileCLIP en-
coders [23], producing multimodal representations. The embeddings, along with
the raw text queries, are then processed by the Unified Clipping Algorithm
(Sec. 4.2), which retrieves relevant results from the multimodal database. Finally,
the retrieved items are re-ranked and displayed through a unified web interface
(Sec. 4.3).

4.1 Data Preprocessing Pipeline

Dynamic-Aware Keyframe Extraction Deep learning-based methods such
as AutoShot [28] achieve accurate shot detection but require substantial com-
putation. Moreover, in highly dynamic scenes, AutoShot often selects only the
last frame, failing to capture full motion variation. To address this, we pro-
pose Dynamic-Aware Keyframe Extraction (DAKE), a lightweight, training-free
method that leverages JPEG file sizes as motion indicators. Dynamic scenes

                U-CESE: Unified Clip-based Event Search Engine                                 5

                                              ……
                                            …

                MobileClip                                                   VisualDB

L21_V001                                          …
                                                …
                                                Visual
                                            Embeddings

          DAKE

                                                                      ReCap  Two news anchors
                                                                               begin the news
                                                                                  broadcast.

K20_V031        Keyframes                   Chào mừng quý vị                 Captions
                       whisper                 và các bạn…
 Videos
                                            Subtitles

                                                                             TextualDB

                Fig. 2. Our data preprocessing pipeline

with large motion, texture, or lighting changes exhibit abrupt variations in com-
pressed file size [3], while static scenes produce stable sizes. Fig. 6 visualizes
normalized JPEG sizes across frame indices, where stable regions correspond to
static scenes and sharp changes denote transitions or dynamic content. Given
two frames fi and fj with corresponding JPEG file sizes si and sj, we define the
steepness S(i, j) as the normalized rate of change in frame size, computed as:

                                100 ×       sj −si
                                            smax
                S(i, j) =                                                ,
                                                                      2
                                (j − i)2 +                    sj −si
                                            100 ×             smax

where smax is the maximum JPEG file size in the video. This normalization
ensures that steepness values remain scale-invariant across videos with varying
resolutions and compression levels. To detect visual transitions, we aggregate
local steepness scores within short temporal windows and select frames with the
highest aggregated values, as described in Algorithm 1. This approach identifies
frames exhibiting strong temporal variations in compression complexity, effec-
tively capturing shot boundaries and dynamic scenes without any learned model.
Moreover, DAKE allows explicit control over the number of detected keyframes,
facilitating experiments on the trade-off between keyframe recall and storage
efficiency. A detailed analysis of DAKE is provided in Sec. 5.1.

6   Nhuan et al.

Algorithm 1 Dynamic-aware Keyframe Selection

Require: List of frame sizes {s1, s2, . . . , sn}, keyframe ratio ρ

Ensure: List of selected keyframe indices K

1: smax ← max(s1, s2, . . . , sn)

2: Slist ← [ ]

3: for i = 1 to n − 1 do

4: sum ← 0

5: count ← 0

6: for j = i + 1 to min(n, i + 3) do

7:  sum ← sum + S(i, j)

8:  count ← count + 1

9: end for

10: steepness ← sum/count

11: Append (i, steepness) to Slist

12: end for

13: Sort Slist in descending order by steepness value
14: k ← ⌊ρ × |Slist|⌋

15: K ← indices of top-k entries in Slist

16: return K

Keyframe Captioning We aim to generate detailed and context-aware cap-
tions for individual keyframes using a Large Vision–Language Model (LVLM)
that incorporates surrounding visual and textual context. For each target keyframe
It, the model receives three inputs: (1) a context window of k preceding and suc-
ceeding keyframes {It−k, . . . , It−1, It, It+1, . . . , It+k}; (2) the subtitle transcript
for the same segment; and (3) the target keyframe itself. The context frames and
subtitle provide auxiliary cues to interpret It but are not directly referenced in
the final caption. Guided by a unified instruction prompt adapted from CESE’s
multi-prompt framework [15], the LVLM integrates reasoning and caption gen-
eration within a single prompt. The instruction directs the model to:

 1. Infer contextual background from the context window and subtitle.
 2. Analyze visual content from entities, attributes, and interactions in It.
 3. Conduct internal reasoning via self-generated visual question–answer pairs

     to ensure descriptive completeness.

    The model outputs a detailed paragraph Ct summarizing all important visual
and contextual cues:

                            Ct = LVLM({It−k:t+k}, St−k:t+k, It),

where St−k:t+k denotes the subtitle segment. This formulation yields fine-grained,
contextually grounded captions focused on the visual specificity of each keyframe.
In our implementation, we employ Gemini [5] as the underlying LVLM.

Recurrence Captioning ReCap (Fig. 3) extends keyframe captioning to
shot-level captioning with temporal consistency. A video is divided into shots:

                                      S = {S0, S1, . . . , ST },

                  U-CESE: Unified Clip-based Event Search Engine                    7

C0            C1                         Ct                         Ct+1

LVLM      M0  . . . LVLM  Mt-1  LVLM         Mt          LVLM             Mt+1 ...

S0            S1                         St                         St+1

      Ct                                     Video Shot Caption

Mt-1  LVLM    Mt  ＝  Memory of Previous          Large Vision             Memory of Previous and
                          Video Shots         Language Model                Current Video Shots

      St                                     Keyframes + Subtitles

                                             Video Shot

      Fig. 3. Overview of the Recurrence Captioning (ReCap) framework.

where each shot St includes keyframes and a subtitle. We utilize AutoShot [28]
for this task. At time step t, the system maintains a memory string Mt capturing
accumulated contextual information. We employ Gemini [5] as the reasoning and
generation engine:

                                 (Ct, Mt) = fLVLM(St, Mt−1),

where fLVLM denotes the LVLM’s reasoning and generation function. This re-
current design mirrors an RNN’s hidden state, preserving contextual continuity
and reflecting evolving narratives across shots. The LVLM selectively retains rel-
evant details (e.g., setting, characters, actions) while discarding outdated ones,
enabling coherent captioning across scene transitions. By combining reasoning
capabilities with recurrent memory, ReCap achieves temporally consistent and
semantically rich video shot descriptions, bridging static image captioning and
dynamic video understanding. Detailed analysis can be found in Sec 5.2.

Overall Data Preprocess Pipeline Fig. 2 illustrates our data preprocessing
pipeline. Following [15], we first extract keyframes using DAKE. Visual embed-
dings are generated by MobileCLIP’s vision encoder [23] and indexed into the
VisualDB by Milvus [24]. OpenAI’s Whisper [21] transcribes audio into subti-
tles, which are combined with keyframes to create detailed captions using ReCap.
Text data are indexed in raw form with Elasticsearch [1] and as embeddings with
MobileCLIP’s textual encoder in Milvus, jointly forming the TextualDB.

8  Nhuan et al.

4.2 Unified Clipping Algorithm

Similar to CESE [15], we retrieve relevant clips instead of individual frames,
referring to the results as "suggestions." Unlike CESE, which separates modality-
specific components with distinct interfaces and ranking schemes, our Unified
Clipping Algorithm integrates all modalities into a single retrieval and ranking
pipeline, eliminating interface switching for a consistent user experience. The
algorithm jointly retrieves timestamps from VisualDB (frame embeddings) and
TextualDB (caption embeddings and raw text), and merges them into a single
list. Given a maximum clip length T , a linear-time two-pointer sweep forms clips
satisfying end − start ≤ T . With fixed T , suggestions generation and ranking
times only scale linearly with the number of retrieved items N . Suggestions are
ranked by the number of unique query IDs covered, with ties broken by the
highest per-frame similarity. The full procedure is shown in Algorithm 2.

4.3 User Interface

We developed a unified user interface that integrates Python, HTML, CSS, and
JavaScript. As shown in Fig. 4, users can input text queries or upload images
for visual search. The interface allows users to view the current question, tog-
gle active modalities, and submit their queries seamlessly. Upon submission, the
system retrieves and displays a ranked list of suggestions, with each row rep-
resenting one retrieved result and its associated frames. By selecting a frame,
users can open an interactive viewer (Fig. 5) that presents detailed informa-
tion about the retrieved frames and corresponding texts. This viewer includes a
built-in video player showing the current frame index in the top-right corner—an
especially useful feature for TRAKE tasks that involve identifying multiple key
moments in a video. Additionally, users can edit their answers in the Answer
field and export the results as a CSV file using the download button. We also
include several keyboard shortcuts to speed up common actions. For example,
pressing Tab automatically appends the current timestamp to the active answer
field. This is especially useful for TRAKE tasks, where users need to quickly
select the exact frame corresponding to multiple events within a single query.

                       U-CESE: Unified Clip-based Event Search Engine       9

Algorithm 2 Unified Clipping Algorithm

Require: Queries Q = {q1, . . . , qn};
           UseFrameEmb[i], UseTextEmb[i], UseTextRaw[i] for each qi;
           retrieval budgets kvis, ktext_emb, ktext_raw;
           maximum suggestion length T ; number of suggestions to return K

Ensure: Top-K suggestions across all videos

1: function RetrieveAll(Q)
2: Rvis ← VisualDB.retrieve_embedding(Q, kvis, UseFrameEmb)
3: Rtext_emb ← TextualDB.retrieve_embedding(Q, ktext_emb, UseTextEmb)
4: Rtext_raw ← TextualDB.retrieve_raw(Q, ktext_raw, UseTextRaw)
5: R ← Flatten(Rvis ∪ Rtext_emb ∪ Rtext_raw) ▷ R is a list of RetrievedFrame
6: return SortBy(R, ⟨video_name, timestamp⟩)
7: end function

8: function Better(Sa, Sb)                                  ▷ two suggestions

9: Ua ← {query_id(f ) : f ∈ Sa.retrieved_frames}

10: Ub ← {query_id(f ) : f ∈ Sb.retrieved_frames}

11: if Ua ̸= Ub then return Ua > Ub

12: end if

13:  ma ← maxf∈Sa.retrieved_frames score(f )

14:  mb ← maxf∈Sb.retrieved_frames score(f )

15: return ma > mb

16: end function

17: function UnifiedClipping(Q, T, K)

18: R ← RetrieveAll(Q)

19: G ← GroupBy(R, video_name)                     ▷ per-video sorted lists

20: S ← [ ]                                        ▷ candidate suggestions

21: for all ⟨v, L⟩ ∈ G do                          ▷ L is sorted by timestamp

22:  l←1

23:  for r ← 1 to |L| do

24:         while l ≤ r and L[r].timestamp − L[l].timestamp > T do

25:          l ← l+1

26:         end while

27:         S ← Suggestion(video_name = v,

             start = L[l].timestamp, end = L[r].timestamp,

             retrieved_frames = L[l:r])

28:         S .append(S )

29:  end for

30: end for

31: S ← Sort(S, Better)

32: return S[1:K]

33: end function

10  Nhuan et al.

                                                    Suggestion

    Query Input

    Fig. 4. Main screen of U-CESE’s user interface

                                Fig. 5. U-CESE’s Interactive Window

5 Ablation Study

5.1 Comparing DAKE with AutoShot

We compare keyframes detected by DAKE with those identified by AutoShot on
the video K01_V001 from the organizers’ dataset. As shown in Fig. 7, setting
ρ = 0.02 (i.e., selecting only 2% of all frames) captures over 80% of AutoShot’s
detections. Let f ps denote the frame rate of the video, and define a match
between a DAKE detection xˆ and an AutoShot detection x˜ if xˆ − ∆ ≤ x˜ ≤ xˆ.
With ρ = 0.02 and ∆ = 0.5 × f ps (half a second), every AutoShot detection
has at least one corresponding DAKE detection. Therefore, we adopt ρ = 0.02
and additionally enforce that at least one keyframe is included within every
∆ = 2 × f ps-frame window for a near-perfect recall.

    Fig. 6 illustrates all methods’ detections for a representative clip. DAKE
consistently covers nearly all AutoShot detections and, between frames 18400

     U-CESE: Unified Clip-based Event Search Engine                                                              11

1.0  Non-keyframe                                                                                        1.0
     True Positive
     False Positive
0.8  False Negative
Normalized JPEG File Size                                                                                0.8
                                                                               AutoShot Detection Ratio
0.6                                                                                                      0.6

0.4                                                                                                      0.4

0.2                                                                                                      0.2                    = 0 (Exact Match)
 17950 18000 18050 18100 18150 F1r8a2m00e1I8n2d5e0x18300 18350 18400 18450 18500                                                = 0.5 × fps
                                                                                                         0.0                    = 1.0 × fps
                                                                                                           0.00                 = 2.0 × fps
                                                                                                                 0.01 0.02 0.03 0.04
                                                                                                                     Keyframe Ratio

Fig. 6. Frame JPEG sizes across frame indices
in a video sample. “True Positive” denotes exact
matches, “False Positive” refers to DAKE detec- Fig. 7. Coverage ratio of AutoShot
tions not found in AutoShot results, and “False detections under varying keyframe
Negative” indicates keyframes detected by Au- ratios ρ and error windows ∆.
toShot but missed by DAKE.

and 18450, captures additional keyframes in highly dynamic regions. This be-
havior results in higher recall and mitigates the risk of losing important visual
information. Since both recall and storage depend on ρ, the trade-off between
them can be efficiently adjusted by tuning this hyperparameter. These results
demonstrate DAKE’s potential as a robust and efficient shot boundary detec-
tion method, particularly suited for applications where storage and processing
efficiency are critical.

12  Nhuan et al.

5.2 Effect of ReCap on Captions

We compare two configurations: (i) No-memory Captioning, which generates cap-
tions for each shot independently using only keyframes and subtitles, and (ii)
our Recurrence Captioning (ReCap), which conditions on the recurrent memory
Mt−1 accumulated from preceding shots. All prompts, keyframe selections, and
subtitle inputs remain identical across configurations; the only difference is the
inclusion of Mt−1. As illustrated in Fig. 8, incorporating recurrent memory en-
ables the model to maintain temporal consistency and capture richer contextual
dependencies across shots.

    Without memory, the model outputs Cˆt, focusing primarily on surface-level
attributes such as clothing and gestures, thereby describing individuals as generic
participants. In contrast, ReCap generates Ct by leveraging historical context
stored in Mt−1 to construct a coherent and temporally grounded narrative. It
correctly identifies the right-hand speaker as the contestant presenting his paint-
ing, provides relevant personal details mentioned in previous shots, and links the
artwork to its title.

    These observations demonstrate ReCap’s effectiveness in enhancing both
identity tracking and contextual coherence. The recurrent memory preserves
entity-specific information, preventing the model from reverting to ambiguous
or generic references. Furthermore, it enables recovery of background context
(e.g., program name, location, or event type) that may not be visually evident
in the current frame, thereby reducing ambiguity and producing more grounded
and informative captions.

            Frame ������                       No-memory   Two young men are prominently featured, standing
                                           Captioning  among several paintings displayed on wooden easels.
#Program: HTV9 '60 giây' news.             Recurrence  The man on the left, wearing a red short-sleeved t-shirt
#Topic: Art competition 'Thế giới          Captioning  and a camouflage baseball cap with sunglasses perched
quanh tôi' for artists with disabilities.              on top, attentively listens to the other individual.
#Location: Exhibition Hall.                  (ReCap)   The man on the right, distinguished by his dark hair,
#Artists_Featured:                                     glasses, red polo shirt with white trim, and black pants
1. char:Bùi-Thị-Thơm: age 23, from                     with white stripes, is actively gesturing with both hands,
Nam Định, paralyzed, paints with                       palms open, seemingly explaining or presenting
mouth (previous focus).                                information about the artworks.
2. char:Lại-Thanh-Quang: age 19,
from Thái Bình (new focus).                                              Caption ���෡���������
#Artworks:
-prop:window-painting: By char:Bùi-                    Two young men are standing in front of several paintings
Thị-Thơm (seen earlier).                               displayed on wooden easels.
-prop:tam-coc-painting:Title "Nắng                     On the right, a man, presumably Lại Thanh Quang, the
chiều tam cốc“(Afternoon Sun in                        19-year-old artist from Thái Bình, is gesturing with both
Tam Cốc) by char:-Lại-Thanh-                           hands as he explains or presents the large artwork before
Quang. Depicts a vibrant river                         him. He is dressed in a vibrant red polo shirt with white
scene with many sampans.                               trim on the collar and sleeves, paired with dark pants and
                                                       glasses, conveying an earnest and articulate demeanor.
       Memory ������������−1                                   To his left, another man, wearing a casual red t-shirt and a
                                                       camouflage baseball cap, listens intently, looking towards
                                                       the speaker with a subtle smile.
                                                       This artwork, identified as "Nắng chiều tam cốc"
                                                       (Afternoon Sun in Tam Cốc)

                                                                         Caption ������������

    Fig. 8. Effect of Recurrent Memory on captions.

U-CESE: Unified Clip-based Event Search Engine  13

5.3 Impact of the Unified Clipping Algorithm for Event Retrieval

In this section, we demonstrate the impact of the Unified Clipping Algorithm
on U-CESE’s retrieval pipeline. Specifically, users can submit multiple queries
simultaneously, each corresponding to a different data modality. As illustrated in
Fig. 4, users can choose between the frame modality for visual descriptive queries
and the text modality for keyword-based search (e.g., query-p2-5-kis). The
selection is performed through the frame and text icons highlighted in red. The
system then generates suggestions based on multimodal retrieval items, including
frames, captions, and subtitles, which are aggregated into a unified interface.
When users open the a suggestion’s interactive window, as shown in Fig. 5, all
related retrieval results are displayed cohesively, allowing seamless exploration
of semantic relationships across modalities. This unified visualization encourages
cross-modal comprehension, and enables users to identify contextually relevant
events spanning visual and textual domains. Consequently, the Unified Clipping
Algorithm not only enhances retrieval accuracy but also improves user experience
by delivering an integrated and interpretable multimodal representation.

5.4 Performance in the final round of the challenge

Our team, Nomial, achieved an excellent overall performance in the final round of
the AIC 2025 [7]. In particular, we obtained outstanding results in TRAKE and
QA tasks, both of which demand a comprehensive understanding of the video
dataset. In these tasks, our Unified Clipping Algorithm demonstrated its effec-
tiveness by jointly leveraging visual and textual modalities to accurately identify
the target clip. Furthermore, the inclusion of keyboard shortcuts enhances the
efficiency of the TRAKE query workflow, as shown in Fig. 9.

6 Conclusion

In this paper, we present U-CESE, a Unified Clip-based Event Search En-
gine developed for the AI Challenge HCMC 2025. Building upon the CESE
framework, U-CESE addresses the limitations of modular inconsistency and in-
efficiency by integrating all modalities into a single cohesive retrieval pipeline.
Our proposed Unified Clipping Algorithm harmonizes the clipping process
across different query types, ensuring both scalability and consistency in event
retrieval. To further enhance efficiency, we introduce DAKE, a training-free
keyframe extraction method leveraging JPEG file size variance to eliminate the
need for complex shot boundary models. Additionally, our ReCap framework
generates temporally consistent and context-aware captions, enriching the se-
mantic understanding of video content. Together, these components enable U-
CESE to achieve contextually rich, effective, and scalable multimodal video re-
trieval across diverse, large-scale datasets. We believe that U-CESE establishes
a robust foundation for future research in unified event search systems, paving
the way for more advanced and intelligent video retrieval solutions.

14  Nhuan et al.

                  Submit

                  (Ctrl-Enter)

                                Choose       Add
                                Suggestion   Timestamp

                                (Click on    (Click on frame
                                suggestion)  + Tab to add
                                             timestamp)

Fig. 9. User’s workflow for TRAKE queries. The chosen query is the 4-th TRAKE in
the final round of AIC2025, which is "In a stir-fried beef cooking video, identify the
first moments when each of the following ingredients makes contact with the pan: E1:
cooking oil, E2: beef, E3: onion, E4: sesbania flower." By utilizing the Tab shortcut,
user can quickly edit the answer.

Acknowledgments

This research is supported by research funding from Faculty of Information Tech-
nology, University of Science, Vietnam National University - Ho Chi Minh City.

References

 1. Elasticsearch: The official distributed search & analytics engine. https://www.
     elastic.co/elasticsearch

 2. Bautista, D., Atienza, R.: Scene text recognition with permuted autoregressive se-
     quence models. In: European conference on computer vision. pp. 178–196. Springer
     (2022)

 3. Boreczky, J., Rowe, L.: Comparison of video shot boundary detection techniques.
     Proceedings of SPIE - The International Society for Optical Engineering 2670 (03
     1996). https://doi.org/10.1117/12.238675

 4. Chen, Z., Wu, J., Wang, W., Su, W., Chen, G., Xing, S., Zhong, M., Zhang, Q.,
     Zhu, X., Lu, L., et al.: Internvl: Scaling up vision foundation models and aligning
     for generic visual-linguistic tasks. In: Proceedings of the IEEE/CVF conference on
     computer vision and pattern recognition. pp. 24185–24198 (2024)

 5. Comanici, G., Bieber, E., Schaekermann, M., Pasupat, I., Sachdeva, N., Dhillon, I.,
     Blistein, M., Ram, O., Zhang, D., Rosen, E., Marris, L., Petulla, S., et al.: Gemini
     2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and
     next generation agentic capabilities (2025), https://arxiv.org/abs/2507.06261

 6. Dinh-Thi, X.B., Dao, A., Trinh, Q.B., Dinh, N.T., Vu, H.N.: Transforming video
     search: leveraging multimodal techniques and llms for optimal retrieval. In: Inter-
     national Symposium on Information and Communication Technology. pp. 121–131.
     Springer (2024)

U-CESE: Unified Clip-based Event Search Engine  15

 7. Do, T.L., Huynh, V.T., Nguyen, H.D., Nguyen-Quang, T., Tran, M.K., Nguyen,
     T.T., Ninh, T.V., Le, T.K., Ngo, T.D., Dang-Nguyen, D.T., Ngo, T.T., Schöffmann,
     K., Gurrin, C., Tran, M.T.: Toward abstraction-level event retrieval in large video
     collections: Leveraging human knowledge and LLM-based reasoning in the Ho Chi
     Minh City AI Challenge 2025. In: Proceedings of the 14th International Symposium
     on Information and Communication Technology (SOICT 2025). CCIS, Springer,
     Nha Trang, Vietnam (2025)

 8. Do, T.L., Nguyen, H.D., Nguyen, Q.T., Tran, M.K., Huynh, V.T., Gurrin, C.,
     Ninh, T.V., Le, T.K., Ngo, T.D., Ngo, T.T., et al.: News event retrieval from
     large video collection in ho chi minh city ai challenge 2023. In: Proceedings of the
     12th International Symposium on Information and Communication Technology.
     pp. 1011–1017 (2023)

 9. Doan, K.T., Huynh, B.G., Hoang, D.T., Pham, T.D., Pham, N.H., Nguyen, Q.,
     Vo, B.Q., Hoang, S.N.: Vintern-1b: an efficient multimodal large language model
     for vietnamese. arXiv preprint arXiv:2408.12480 (2024)

10. Douze, M., Guzhva, A., Deng, C., Johnson, J., Szilvasy, G., Mazaré, P.E., Lomeli,
     M., Hosseini, L., Jégou, H.: The faiss library. IEEE Transactions on Big Data
     (2025)

11. Elman, J.L.: Finding structure in time. Cognitive Science 14(2), 179–211 (1990)
12. Gia, B.T., Khanh, T.B.C., Thanh, T.L.T., Tran, K., Trong, H.H., Doan, T.T., Le,

     K., Do, T., Le, D.D., Ngo, T.D.: Addressing ambiguous queries in video retrieval
     with advanced temporal search. In: International Symposium on Information and
     Communication Technology. pp. 167–180. Springer (2024)
13. Google: Faster r-cnn inception resnet v2 model. https://www.kaggle.
     com/models/google/faster-rcnn-inception-resnet-v2/tensorFlow1/
     faster-rcnn-openimages-v4-inception-resnet-v2 (2024)
14. Gurrin, C., Jónsson, B.Þ., Nguyen, D.T.D., Healy, G., Lokoc, J., Zhou, L., Rossetto,
     L., Tran, M.T., Hürst, W., Bailer, W., et al.: Introduction to the sixth annual lifelog
     search challenge, lsc’23. In: Proceedings of the 2023 ACM International Conference
     on Multimedia Retrieval. pp. 678–679 (2023)
15. Le, D.N., Nguyen, H.P., Lam, T.D., Dang, M.N., Le, M.H.: Cese: A clip-based event
     search engine for ai challenge hcmc 2024. In: Buntine, W., Fjeld, M., Tran, T., Tran,
     M.T., Huynh Thi Thanh, B., Miyoshi, T. (eds.) Information and Communication
     Technology. pp. 254–267. Springer Nature Singapore, Singapore (2025)
16. Li, J., Li, D., Savarese, S., Hoi, S.: Blip-2: Bootstrapping language-image pre-
     training with frozen image encoders and large language models. In: International
     conference on machine learning. pp. 19730–19742. PMLR (2023)
17. Lokoč, J., Vesely`, P., Mejzlík, F., Kovalčík, G., Souček, T., Rossetto, L., Schoeff-
     mann, K., Bailer, W., Gurrin, C., Sauter, L., et al.: Is the reign of interactive search
     eternal? findings from the video browser showdown 2020. ACM Transactions on
     Multimedia Computing, Communications, and Applications (TOMM) 17(3), 1–26
     (2021)
18. Phat, T.A., Minh, T.T., Hoan, D.N.T., Nguyen, K.D.: Revimm: Enhanced video
     retrieval with reweighting mechanism for multi-modal queries. In: International
     Symposium on Information and Communication Technology. pp. 18–28. Springer
     (2024)
19. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G.,
     Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from
     natural language supervision. In: International conference on machine learning. pp.
     8748–8763. PmLR (2021)

16  Nhuan et al.

20. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G.,
     Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from
     natural language supervision. In: International conference on machine learning. pp.
     8748–8763. PMLR (2021)

21. Radford, A., Kim, J.W., Xu, T., Brockman, G., McLeavey, C., Sutskever, I.: Robust
     speech recognition via large-scale weak supervision (2022), https://arxiv.org/
     abs/2212.04356

22. Souček, T., Lokoč, J.: Transnet v2: An effective deep network architecture for fast
     shot transition detection. arXiv preprint arXiv:2008.04838 (2020)

23. Vasu, P.K.A., Pouransari, H., Faghri, F., Vemulapalli, R., Tuzel, O.: Mobileclip:
     Fast image-text models through multi-modal reinforced training. In: Proceedings of
     the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
     (June 2024)

24. Wang, J., Yi, X., Guo, R., Jin, H., Xu, P., Li, S., Wang, X., Guo, X., Li, C., Xu, X.,
     et al.: Milvus: A purpose-built vector data management system. In: Proceedings of
     the 2021 International Conference on Management of Data. pp. 2614–2627 (2021)

25. Wang, W., Bao, H., Dong, L., Bjorck, J., Peng, Z., Liu, Q., Aggarwal, K., Mo-
     hammed, O.K., Singhal, S., Som, S., et al.: Image as a foreign language: Beit pre-
     training for vision and vision-language tasks. In: Proceedings of the IEEE/CVF
     Conference on Computer Vision and Pattern Recognition. pp. 19175–19186 (2023)

26. Ye, M., Zhang, J., Zhao, S., Liu, J., Liu, T., Du, B., Tao, D.: Deepsolo: Let trans-
     former decoder with explicit points solo for text spotting. In: Proceedings of the
     IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 19348–
     19357 (2023)

27. Zhu, C., Jia, Q., Chen, W., Guo, Y., Liu, Y.: Deep learning for video-text re-
     trieval: a review. International Journal of Multimedia Information Retrieval 12(1),
     3 (2023)

28. Zhu, W., Huang, Y., Xie, X., Liu, W., Deng, J., Zhang, D., Wang, Z., Liu, J.:
     Autoshot: A short video dataset and state-of-the-art shot boundary detection.
     In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
     Recognition Workshops (CVPRW) (2023)
