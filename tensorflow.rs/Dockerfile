FROM tensorflow/tensorflow:latest-gpu

RUN pip install tifffile ipython ipdb jedi==0.17.2 earthengine-api pandas rasterio imagecodecs  


# Installing Google Cloud SDK
RUN curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-342.0.0-linux-x86_64.tar.gz
RUN mkdir -p /opt/gcloud && mv google-cloud-sdk-342.0.0-linux-x86_64.tar.gz /opt/gcloud/google-cloud-sdk-342.0.0-linux-x86_64.tar.gz
RUN cd /opt/gcloud && gunzip google-cloud-sdk-342.0.0-linux-x86_64.tar.gz && tar -xf google-cloud-sdk-342.0.0-linux-x86_64.tar
