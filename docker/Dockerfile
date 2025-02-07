FROM legacysurvey/legacypipe:DR10.3.1b

RUN git clone https://github.com/nadafalou/LegacyPM.git && echo 4

RUN for x in \
    fireworks \
    ; do pip3 install $x; done \
    && rm -Rf /root/.cache/pip