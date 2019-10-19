FROM minio/minio:RELEASE.2019-08-21T19-40-07Z
RUN apk add tzdata
RUN cp /usr/share/zoneinfo/GMT /etc/localtime
RUN echo "GMT" > /etc/timezone