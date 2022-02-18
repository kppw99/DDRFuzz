# DDRFuzz: Data-driven Fuzzing with Seq2Seq Model for Seed Generation
![image](https://user-images.githubusercontent.com/48042609/154632386-144f2eef-d559-4471-8f8a-b810927bd4a6.png)

## Abstract
![image](https://user-images.githubusercontent.com/48042609/154632653-04dc1ddf-f67c-4fbf-be35-4ac5836fd5ff.png)

## Prerequisite
- **OS:** ubuntu (18.04 LTS)
- **Container:** docker (19.03.6)
- **Fuzzer:** AFL (2.5.b)
- **DATA Collector:** [AFL4DDRFuzz](https://github.com/onsoim/afl4ddrfuzz)

## Description of Directory
*(D: directory / F: file)*
- **[D] projects:** target projects
- **[D] src:** source code of ddrfuzz
- **[F] Dockerfile:** dockerfile for base environment such as os, AFL, utils, etc.
- **[F] docker-compose.yml:** docker compose file for fuzzbuilderex, target libraries
- **[F] build.sh:** script file to build target projects

## Publications
```
DDRFuzz: Data-driven Fuzzing with Seq2Seq Model for Seed Generation

@article{
TBD
}
```

## About
This program is authored and maintained by **Sanghoon(Kevin) Jeon**, **Dongyoung Kim**, and **Minsoo Ryu**.
> Email: kppw99@gmail.com, ehddud758@gmail.com, onsoim@gmail.com

> GitHub[@DDRFuzz](https://github.com/kppw99/ddrfuzz)
