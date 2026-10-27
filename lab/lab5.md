# Pipeline: from git push to a running container

```
    you                          github actions                     your laptop
     |                                 |                                 |
  git push ------------------------->  |                                 |
     |                            [ test ]        automatic              |
     |                             ruff + pytest                         |
     |                                 |                                 |
     |                            [ smoke ]       automatic              |
     |                      docker build, run with SKIP_MODEL=1,         |
     |                            probe /health                          |
     |                                 |                                 |
     |                            [ push ]        automatic, main only   |
     |                    ghcr.io/jadallam/mlops-lab-1:<commit sha>      |
     |                                 |                                 |
     |                                 +--- image is published ---+      |
     |                                                            |      |
  open a PR                                                       |      |
     |  checks must be green            HUMAN: review and merge   |      |
     |                                                            v      |
                                                            docker compose pull / up
                                                                   |
                                                                   v
                                                        container starts, reads
                                                        models:/food11@champion
                                                                   |
                                                                   v
                                                             serving traffic
```

## The model side, which moves on its own

```
  uv run train.py            HUMAN decides what to train
        |
        v
  run logged in mlflow  ->  register_model  ->  version N
        |
        v
  uv run promote.py --version N          automatic gate
        |
        +-- acc >= champion + 2*se  ->  champion alias moves to N
        |
        +-- otherwise               ->  challenger alias set on N
        |
        v
  docker compose restart inference       HUMAN, or a webhook
```

## Where each version is decided

| | decided by | name it travels under |
|---|---|---|
| code | `git push` | commit sha, as the image tag |
| model | `promote.py` | the `champion` alias |

The two never meet in a build. The image has no model in it, and the registry has no code
in it. A container is one commit sha plus whatever `champion` points at when it starts.

## Automatic vs human

automatic: lint, tests, image build, health probe, image push to ghcr, the promotion gate
human: writing the code, opening and merging the PR, starting a training run, restarting
the inference service to pick up a new champion
