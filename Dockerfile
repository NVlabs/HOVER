# syntax=docker.io/docker/dockerfile:1.7-labs
FROM nvcr.io/nvidia/isaac-lab:2.0.0

# Hide conflicting Vulkan files, if needed.
RUN if [ -e "/usr/share/vulkan" ] && [ -e "/etc/vulkan" ]; then \
      mv /usr/share/vulkan /usr/share/vulkan_hidden; \
    fi

# Install vs code server and file browser to interact with the running workflow.
RUN curl -fsSL https://code-server.dev/install.sh | bash
RUN curl -fsSL https://raw.githubusercontent.com/filebrowser/get/master/get.sh | bash

WORKDIR /workspace/neural_wbc

COPY \
  --exclude=Dockerfile \
  --exclude=data/ \
  --exclude=logs/ \
  --exclude=release/ \
  --exclude=run.py \
  --exclude=workflows/ \
  . /workspace/neural_wbc

RUN ./install_deps.sh

# Bake in a resume policy. This could be a teacher or student policy. We set this first since we
# expect it to change less frequently than the policy below.
ARG RESUME_PATH=neural_wbc/data/data/policy/h1:student/
COPY ${RESUME_PATH} /workspace/neural_wbc/policy/resume
ENV RESUME_PATH=/workspace/neural_wbc/policy/resume
ARG RESUME_CHECKPOINT=model_4000.pt
ENV RESUME_CHECKPOINT=${RESUME_CHECKPOINT}

# Bake in the main policy. This could be a teacher or student policy. Depending on the use case this
# is used as the teacher policy or the evaluation policy.
ARG POLICY_PATH=neural_wbc/data/data/policy/h1:teacher/
COPY ${POLICY_PATH} /workspace/neural_wbc/policy/
ENV POLICY_PATH=/workspace/neural_wbc/policy/
ARG POLICY_CHECKPOINT=model_76000.pt
ENV POLICY_CHECKPOINT=${POLICY_CHECKPOINT}
