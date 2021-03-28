FROM ddrfuzz/base
LABEL maintainer="onsoim <onsoim@gmail.com>"


RUN apt-get update && \
    apt-get install -y make autoconf automake libtool curl cmake python llvm-dev libclang-dev clang

RUN git clone --depth 1 --recursive https://github.com/mozilla/mp4parse-rust mp4parse-rust
WORKDIR /tool/mp4parse-rust/mp4parse_capi/fuzz/

RUN curl -sSf https://sh.rustup.rs/ > init.sh && \
    chmod +x init.sh && \
    ./init.sh -y && \
    rm init.sh
ENV PATH="/root/.cargo/bin/:${PATH}"
RUN cargo install cargo-fuzz
RUN rustup install nightly && rustup default nightly
RUN cargo fuzz build -O --debug-assertions

ENV PROJECT_DIR=/tool/mp4parse-rust

# collect avif files
RUN mkdir $PROJECT_DIR/avif_corpus
RUN find $PROJECT_DIR/mp4parse -type f -name '*.avif' -exec cp '{}' $PROJECT_DIR/avif_corpus \;

# collect mp4 files
RUN mkdir $PROJECT_DIR/mp4_corpus
RUN find $PROJECT_DIR/mp4parse/tests -type f -name '*.mp4' -exec cp '{}' $PROJECT_DIR/mp4_corpus \;
RUN find $PROJECT_DIR/mp4parse_capi/tests/ -type f -name '*.mp4' -exec cp '{}' $PROJECT_DIR/mp4_corpus \;


RUN apt-get install -y zip
ENV OUT=$PROJECT_DIR/Fuzzing
RUN mkdir $OUT
ENV FUZZ_TARGET_OUTPUT_DIR=$PROJECT_DIR/mp4parse_capi/fuzz/target/x86_64-unknown-linux-gnu/release
RUN for f in $SRC/mp4parse-rust/mp4parse_capi/fuzz/fuzz_targets/*.rs; do     FUZZ_TARGET_NAME=$(basename ${f%.*});     cp $FUZZ_TARGET_OUTPUT_DIR/$FUZZ_TARGET_NAME $OUT/;     cp $PROJECT_DIR/mp4parse_capi/fuzz/mp4.dict $OUT/$FUZZ_TARGET_NAME.dict;     cp $SRC/default.options $OUT/$FUZZ_TARGET_NAME.options;     zip -jr $OUT/${FUZZ_TARGET_NAME}_seed_corpus.zip $PROJECT_DIR/${FUZZ_TARGET_NAME}_corpus/; done

WORKDIR $OUT