FROM continuumio/miniconda3

WORKDIR /home/app
    
RUN apt-get update && \
    apt-get install -y nano python3.10

COPY requirements_inf.txt requirements_inf.txt
RUN pip install -r requirements_inf.txt
COPY main_inf.py /home/app/

CMD streamlit run --server.port 80 main_inf.py