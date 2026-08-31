import pytest
from src.guardrails.output_validator import validate_output, OutputValidationResult
from src.agent.state import Evidence

def get_sample_evidence():
    return [
        Evidence(
            source_type="pull_request",
            source_id="PR-11049",
            url="https://github.com/octocat/Hello-World/pull/11049",
            content={"title": "Fix something"}
        ),
        Evidence(
            source_type="issue",
            source_id="Issue-12",
            url="https://github.com/octocat/Hello-World/issues/12",
            content={"title": "Bug"}
        ),
        Evidence(
            source_type="file",
            source_id="README.md",
            url="https://github.com/octocat/Hello-World/blob/master/README.md",
            content={"content": "Hello World"}
        )
    ]

def test_valid_pr_citation():
    ev = get_sample_evidence()
    response = "This was fixed in [PR-11049](https://github.com/octocat/Hello-World/pull/11049)."
    res = validate_output(response, ev)
    assert res.valid is True
    assert len(res.validated_citations) == 1
    assert res.validated_citations[0].is_valid is True

def test_valid_issue_citation():
    ev = get_sample_evidence()
    response = "Check out [Issue-12](https://github.com/octocat/Hello-World/issues/12)."
    res = validate_output(response, ev)
    assert res.valid is True

def test_valid_file_citation():
    ev = get_sample_evidence()
    response = "See [README.md](https://github.com/octocat/Hello-World/blob/master/README.md)."
    res = validate_output(response, ev)
    assert res.valid is True

def test_fabricated_pr_citation():
    ev = get_sample_evidence()
    response = "Fixed in [PR-99999](https://github.com/octocat/Hello-World/pull/99999)."
    res = validate_output(response, ev)
    assert res.valid is False
    assert "was not present in retrieved evidence" in res.violations[0]
    
def test_fabricated_issue_citation():
    ev = get_sample_evidence()
    response = "Fixed in [Issue-999](https://github.com/octocat/Hello-World/issues/999)."
    res = validate_output(response, ev)
    assert res.valid is False
    assert "was not present in retrieved evidence" in res.violations[0]

def test_wrong_repository():
    ev = get_sample_evidence()
    response = "Fixed in [PR-1](https://github.com/some-other-user/some-other-repo/pull/1)."
    res = validate_output(response, ev)
    assert res.valid is False
    assert "different repository" in res.violations[0]

def test_mismatched_url():
    ev = get_sample_evidence()
    # Source ID matches, but URL is different
    response = "Fixed in [PR-11049](https://github.com/octocat/Hello-World/pull/999)."
    res = validate_output(response, ev)
    assert res.valid is False
    assert "does not match the evidence URL" in res.violations[0]

def test_missing_citation():
    ev = get_sample_evidence()
    response = "The issue was fixed in PR 11049." # No markdown citation
    res = validate_output(response, ev)
    assert res.valid is False
    assert "contains no citations despite available citable evidence" in res.violations[0]

def test_legitimate_insufficient_evidence_response():
    ev = get_sample_evidence()
    response = "I could not determine the answer from the retrieved evidence."
    res = validate_output(response, ev)
    assert res.valid is True
    assert len(res.violations) == 0

def test_multiple_valid_citations():
    ev = get_sample_evidence()
    response = "According to [PR-11049](https://github.com/octocat/Hello-World/pull/11049) and [Issue-12](https://github.com/octocat/Hello-World/issues/12), it is fixed."
    res = validate_output(response, ev)
    assert res.valid is True
    assert len(res.validated_citations) == 2

def test_mixture_of_valid_and_invalid_citations():
    ev = get_sample_evidence()
    response = "According to [PR-11049](https://github.com/octocat/Hello-World/pull/11049) and [Fabricated](https://github.com/octocat/Hello-World/pull/999)."
    res = validate_output(response, ev)
    assert res.valid is False
    assert len(res.validated_citations) == 2
    assert res.validated_citations[0].is_valid is True
    assert res.validated_citations[1].is_valid is False
    assert len(res.violations) == 1
    
def test_evidence_with_no_url():
    ev = [
        Evidence(
            source_type="local_file",
            source_id="local_id",
            url=None,
            content={"data": "test"}
        )
    ]
    # In this case, LLM shouldn't really be citing a URL, but if it cites something else:
    response = "Based on [local_id](local_url)."
    res = validate_output(response, ev)
    assert res.valid is False # since 'local_url' wasn't in evidence URLs and source_id 'local_id' exists but URLs don't match (None vs local_url)
    assert "does not match the evidence URL" in res.violations[0]

def test_duplicate_evidence():
    ev = get_sample_evidence()
    # Add exact same evidence twice
    ev.append(get_sample_evidence()[0])
    response = "Fixed in [PR-11049](https://github.com/octocat/Hello-World/pull/11049)."
    res = validate_output(response, ev)
    assert res.valid is True
