"""Hugging Face 데이터셋 저장소로 학습 산출물을 동기화한다.

**무엇을 푸는가** — Colab 세션이 끊기면 `/content` 가 통째로 사라진다. 판 안에서
재개하는 기능(`Trainer.try_resume`)은 이미 있지만, **재개할 `last.pt` 가 남아
있어야** 쓸모가 있다. Drive 마운트도 끊기는 마당이라, 학습 도중의 체크포인트를
**저장소 밖**에 두는 길이 필요하다.

그래서 방향이 둘이다. 올리기만으로는 아무것도 안 풀린다.

  * **내려받기(`pull`)** — sweep 을 시작하기 전에 HF 에 있는 것을 먼저 가져온다.
    `summary.json` 이 있는 판은 건너뛰어지고, `last.pt` 가 있는 판은 그 epoch
    에서 이어진다. **기존 재개 논리를 하나도 안 고치고 그대로 쓴다.**
  * **올리기(`push_*`)** — N epoch 마다 `last.pt` + `log.csv`, 판이 끝나면 그 판
    폴더, sweep 이 끝나면 전부.

**올리기 실패가 학습을 죽이면 안 된다.** 30 분짜리 판을 네트워크 딸꾹질로
잃는 것이 원래 막으려던 일이다 — 모든 통신은 예외를 삼키고 경고만 찍는다.

토큰은 `huggingface_hub` 가 알아서 찾는다 (`huggingface-cli login` 이나
`HF_TOKEN` 환경변수). 여기서는 토큰을 인자로 받지도, 찍지도 않는다.
"""
from __future__ import annotations

import shutil
from pathlib import Path

__all__ = ["HFStore"]

# 판마다 올리는 것. `best.pt` 는 추론용이라 판이 끝나면 `summary.json` 으로
# 대체되고, 중간 재개에는 `last.pt` 만 있으면 된다.
EPOCH_FILES = ("last.pt", "log.csv")


class HFStore:
    """`repo_id` 가 없으면 **전부 무동작**이다 — 호출부에 분기를 만들지 않는다."""

    def __init__(self, repo_id: str | None, prefix: str = "sweep",
                 every: int = 10, private: bool = True):
        self.repo_id = repo_id or None
        self.prefix = prefix.strip("/") or "sweep"
        self.every = max(1, int(every))
        self.private = bool(private)
        self._api = None
        self._ready = False

    @property
    def enabled(self) -> bool:
        return self.repo_id is not None

    # ---------------- 내부
    def _hub(self):
        """`huggingface_hub` 를 늦게 부른다 — 안 쓰는 환경에 의존성을 안 만든다."""
        if self._api is None:
            try:
                from huggingface_hub import HfApi
            except ImportError as e:      # 메시지를 분명히 — 이게 제일 흔한 실패다
                raise RuntimeError(
                    "huggingface_hub 가 없다. `pip install -U huggingface_hub` 후 "
                    "`huggingface-cli login` 으로 토큰을 넣어라.") from e
            self._api = HfApi()
        if not self._ready:
            self._api.create_repo(self.repo_id, repo_type="dataset",
                                  private=self.private, exist_ok=True)
            self._ready = True
        return self._api

    def _warn(self, what: str, e: Exception) -> None:
        print(f"  [hf] {what} 실패 — 학습은 계속한다: {type(e).__name__}: {e}",
              flush=True)

    # ---------------- 내려받기
    def pull(self, out_root: Path) -> int:
        """HF 에 있는 이 prefix 의 파일을 `out_root` 로 가져온다.

        **이미 있는 파일은 덮어쓰지 않는다.** 돌고 있는 기계의 것이 최신이고,
        HF 것은 지난 세션의 흔적이다. 반대로 덮으면 이번 세션의 진행을 지운다.
        """
        if not self.enabled:
            return 0
        try:
            from huggingface_hub import snapshot_download
            self._hub()                            # 저장소 보장
            local = snapshot_download(self.repo_id, repo_type="dataset",
                                      allow_patterns=[f"{self.prefix}/**"])
        except Exception as e:
            self._warn("pull", e)
            return 0
        src = Path(local) / self.prefix
        if not src.is_dir():
            print(f"  [hf] {self.repo_id}:{self.prefix} 에 아직 아무것도 없다")
            return 0
        n = 0
        for f in src.rglob("*"):
            if not f.is_file():
                continue
            dst = out_root / f.relative_to(src)
            if dst.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)
            n += 1
        print(f"  [hf] {self.repo_id}:{self.prefix} 에서 {n} 개 가져왔다 "
              f"(이미 있던 것은 그대로 뒀다)")
        return n

    # ---------------- 올리기
    def should_push(self, epoch: int) -> bool:
        return self.enabled and epoch > 0 and epoch % self.every == 0

    def push_epoch(self, run_dir: Path, epoch: int) -> bool:
        """`every` 의 배수인 epoch 에서만 `last.pt` + `log.csv` 를 올린다."""
        if not self.should_push(epoch):
            return False
        try:
            api = self._hub()
            for nm in EPOCH_FILES:
                p = run_dir / nm
                if p.exists():
                    api.upload_file(path_or_fileobj=str(p), repo_id=self.repo_id,
                                    repo_type="dataset",
                                    path_in_repo=f"{self.prefix}/{run_dir.name}/{nm}",
                                    commit_message=f"{run_dir.name} epoch {epoch}")
        except Exception as e:
            self._warn(f"{run_dir.name} epoch {epoch} 올리기", e)
            return False
        print(f"  [hf] {run_dir.name} epoch {epoch} 올렸다", flush=True)
        return True

    def push_run(self, run_dir: Path) -> bool:
        """판 하나가 끝났다. 폴더를 올리고 **죽은 `last.pt` 는 지운다.**

        끝난 판의 `last.pt` 는 재개에 쓰이지 않는데(그 판은 `summary.json` 으로
        건너뛰어진다) 저장소에는 계속 남는다. HF 는 git 이라 이력이 쌓이므로
        판마다 10 MB 대 파일이 남으면 금세 부담이 된다.
        """
        if not self.enabled:
            return False
        try:
            api = self._hub()
            api.upload_folder(folder_path=str(run_dir), repo_id=self.repo_id,
                              repo_type="dataset",
                              path_in_repo=f"{self.prefix}/{run_dir.name}",
                              ignore_patterns=["last.pt"],
                              commit_message=f"{run_dir.name} 완료")
            try:
                api.delete_file(path_in_repo=f"{self.prefix}/{run_dir.name}/last.pt",
                                repo_id=self.repo_id, repo_type="dataset",
                                commit_message=f"{run_dir.name} 끝 — 중간 체크포인트 정리")
            except Exception:
                pass                  # 원래 없었으면 그만이다
        except Exception as e:
            self._warn(f"{run_dir.name} 올리기", e)
            return False
        print(f"  [hf] {run_dir.name} 올렸다", flush=True)
        return True

    def push_all(self, out_root: Path) -> bool:
        """sweep 이 끝났다 — **전체 출력**을 올린다."""
        if not self.enabled:
            return False
        try:
            self._hub().upload_folder(
                folder_path=str(out_root), repo_id=self.repo_id,
                repo_type="dataset", path_in_repo=self.prefix,
                ignore_patterns=["last.pt"],
                commit_message=f"{self.prefix} sweep 완료")
        except Exception as e:
            self._warn("전체 올리기", e)
            return False
        print(f"  [hf] 전체를 {self.repo_id}:{self.prefix} 에 올렸다", flush=True)
        return True
