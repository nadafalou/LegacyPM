FROM legacysurvey/legacypipe:DR10.3.1b

RUN git clone https://github.com/nadafalou/LegacyPM.git && echo 6

RUN for x in \
    fireworks \
    pytest \
    ; do pip3 install $x; done \
    && rm -Rf /root/.cache/pip