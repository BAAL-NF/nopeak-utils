FROM nfcore/base:1.12.1

RUN apt update && apt install -y g++ gcc libcurl4-openssl-dev libz-dev
# Currently installing gimmemotifs through conda, as v0.17.2 is not yest pushed to PyPi
RUN conda install -c conda-forge -c bioconda python=3.8 
RUN pip install poetry
# Workaround for pip 10 issue: https://stackoverflow.com/questions/63383400/error-cannot-uninstall-ruamel-yaml-while-creating-docker-image-for-azure-ml-a
RUN pip install --ignore-installed ruamel-yaml==0.17.21
COPY nopeak_utils pyproject.toml /build/
WORKDIR /build/
RUN --mount=type=secret,id=gitlab,dst=/etc/credentials.sh\
    . /etc/credentials.sh &&\
    POETRY_VIRTUALENVS_CREATE=false poetry install
ENV PYTHONPATH "${PYTHONPATH}:/build"