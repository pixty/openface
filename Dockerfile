FROM bamos/openface:latest

RUN cd ~/openface && \
    pip2 install flask

COPY ./api /root/api

WORKDIR /root/api

EXPOSE 8000 9000 5000

CMD /bin/bash -l -c '/root/api/start.sh'
