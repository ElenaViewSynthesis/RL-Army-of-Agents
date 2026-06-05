"""Typed models for git tools."""

from pydantic import BaseModel, Field


class GitResult(BaseModel):
    success: bool = True
    error: str | None = None


class GitStatusInput(BaseModel):
    repo: str = "."


class GitStatusOutput(GitResult):
    repo: str
    branch: str
    is_clean: bool
    output: str


class GitDiffInput(BaseModel):
    repo: str = "."
    staged: bool = False
    path: str | None = None


class GitDiffOutput(GitResult):
    repo: str
    diff: str


class GitCommitSummary(BaseModel):
    sha: str
    message: str


class GitLogInput(BaseModel):
    repo: str = "."
    max_count: int = Field(default=10, ge=1, le=100)


class GitLogOutput(GitResult):
    repo: str
    commits: list[GitCommitSummary]


class GitBlameInput(BaseModel):
    repo: str = "."
    path: str = Field(min_length=1)
    start_line: int | None = Field(default=None, ge=1)
    end_line: int | None = Field(default=None, ge=1)


class GitBlameLine(BaseModel):
    commit: str
    line_number: int = Field(ge=1)
    text: str


class GitBlameOutput(GitResult):
    repo: str
    path: str
    lines: list[GitBlameLine]


class GitBranchCreateInput(BaseModel):
    repo: str = "."
    name: str = Field(min_length=1)
    checkout: bool = True


class GitBranchCreateOutput(GitResult):
    repo: str
    name: str
    checked_out: bool


class GitBranchListInput(BaseModel):
    repo: str = "."


class GitBranchListOutput(GitResult):
    repo: str
    branches: list[str]
    current: str | None = None


class GitCheckoutInput(BaseModel):
    repo: str = "."
    ref: str = Field(min_length=1)


class GitCheckoutOutput(GitResult):
    repo: str
    ref: str
    output: str


class GitCommitInput(BaseModel):
    repo: str = "."
    message: str = Field(min_length=1)
    all_changes: bool = False


class GitCommitOutput(GitResult):
    repo: str
    commit_sha: str | None = None
    output: str


class GitStashInput(BaseModel):
    repo: str = "."
    message: str | None = None


class GitStashOutput(GitResult):
    repo: str
    output: str


class GitShowCommitInput(BaseModel):
    repo: str = "."
    ref: str = "HEAD"


class GitShowCommitOutput(GitResult):
    repo: str
    ref: str
    output: str


class GitListChangedFilesInput(BaseModel):
    repo: str = "."
    staged: bool = False


class GitListChangedFilesOutput(GitResult):
    repo: str
    files: list[str]


class GitTagInput(BaseModel):
    repo: str = "."
    name: str = Field(min_length=1)
    ref: str = "HEAD"
    message: str | None = None


class GitTagOutput(GitResult):
    repo: str
    name: str
    output: str
