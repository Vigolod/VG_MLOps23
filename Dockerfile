FROM nvidia/cuda:12.3.0-devel-ubuntu22.04
RUN apt-get -y update
RUN apt-get install -y python3.10
RUN apt-get install -y python3.10-venv
RUN apt-get install -y python3-pip
RUN python3 -m pip install pipx && python3 -m pipx ensurepath
RUN pipx install poetry

ENV PATH="${PATH}:/root/.local/bin"
ENV PATH="${PATH}:/root/.poetry/bin"
ENV PROJECT_PATH="/vg_mlops"
ENV CONF_PATH="${PROJECT_PATH}/conf" \
    DATA_PATH="${PROJECT_PATH}/data"

RUN mkdir -p ${CONF_PATH} && mkdir -p ${DATA_PATH}
COPY conf ${CONF_PATH}/
COPY model.py train.py infer.py pyproject.toml ${PROJECT_PATH}/
WORKDIR $PROJECT_PATH
RUN poetry install
