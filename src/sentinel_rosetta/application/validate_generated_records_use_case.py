"""Pure logic: counts records through injected repositories."""

from sentinel_rosetta.domain.interfaces import GeneratedRecordRepository


class ValidateGeneratedRecordsUseCase:
    def __init__(
        self, record_repositories: tuple[GeneratedRecordRepository, ...]
    ) -> None:
        self._record_repositories = record_repositories

    def execute(self) -> int:
        record_count = 0
        for record_repository in self._record_repositories:
            record_count += len(record_repository.load_all())
        return record_count
