FROM tensorflow/tensorflow:latest-gpu

RUN pip install tifffile
RUN pip install ipython
RUN pip install jedi==0.17.2
RUN pip install earthengine-api 
RUN pip install pandas
RUN pip install rasterio
RUN pip install imagecodecs  


# Installing Google Cloud SDK
RUN curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-342.0.0-linux-x86_64.tar.gz
RUN mkdir -p /opt/gcloud && mv google-cloud-sdk-342.0.0-linux-x86_64.tar.gz /opt/gcloud/google-cloud-sdk-342.0.0-linux-x86_64.tar.gz
RUN cd /opt/gcloud && gunzip google-cloud-sdk-342.0.0-linux-x86_64.tar.gz && tar -xf google-cloud-sdk-342.0.0-linux-x86_64.tar
