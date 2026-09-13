# Track decision

I am committing to **Track B — LLM/RAG (Arabic Legal Document Q&A)** for this final project.
The deciding constraint is GPU access: my only machine right now is a laptop (Lenovo LOQ
15IAX9) with a modest mobile GPU, not a workstation-class card, and I don't have a Colab Pro
subscription or a standing Kaggle GPU quota lined up for this course's timeline — while my other
active work (an anti-drone multi-agent system, a person-following robot) already competes for
whatever GPU time I do have. Track A's optimization module leans hard on TensorRT and
distillation runs that assume reliable, repeated GPU access, and treating Colab as a documented
workaround for every GPU-gated step would cost more friction than it's worth across five modules.
Track B is built to run end-to-end on CPU-served or API-backed models, so nothing in the critical
path is gated on hardware I'd have to borrow or wait for — and it pushes me into RAG evaluation,
cost/token observability, and Arabic document handling, which is a genuinely different skill set
from the classical supervised-learning and ROS2 vision work that already makes up most of my
portfolio, rather than a repeat of it.

**I am committing to Track B because it removes my one real infrastructure constraint (GPU
access) while adding the applied-AI skills my existing project history doesn't yet cover.**
