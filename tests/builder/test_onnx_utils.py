"""Tests for ONNX graph operations -- subgraph extraction and splitting."""

from __future__ import annotations

import pytest


@pytest.mark.cpu
class TestExtractSubgraph:
    """Tests for extract_subgraph()."""

    @pytest.mark.xfail(
        reason="Source bug: onnx_graphsurgeon Variable objects are not hashable in newer versions",
        strict=False,
    )
    def test_extract_basic(self, onnx_path):
        """Extract a subgraph from a valid ONNX model."""
        import onnx

        from trtutils.builder.onnx import extract_subgraph

        model = onnx.load(str(onnx_path))
        num_nodes = len(model.graph.node)
        if num_nodes < 2:
            pytest.skip("Model has too few nodes for subgraph test")

        subgraph = extract_subgraph(model, 0, num_nodes - 1)
        assert isinstance(subgraph, onnx.ModelProto)

    def test_extract_invalid_start(self, onnx_path):
        """ValueError for negative start_idx."""
        import onnx

        from trtutils.builder.onnx import extract_subgraph

        model = onnx.load(str(onnx_path))
        with pytest.raises(ValueError, match="start_idx must be greater"):
            extract_subgraph(model, -1, 0)

    def test_extract_start_gt_end(self, onnx_path):
        """ValueError when start_idx > end_idx."""
        import onnx

        from trtutils.builder.onnx import extract_subgraph

        model = onnx.load(str(onnx_path))
        num_nodes = len(model.graph.node)
        if num_nodes < 2:
            pytest.skip("Model has too few nodes")
        # Use indices within range to hit start > end check
        with pytest.raises(ValueError, match="start_idx must be less"):
            extract_subgraph(model, num_nodes - 1, 0)

    def test_extract_end_too_large(self, onnx_path):
        """ValueError when end_idx >= num_nodes."""
        import onnx

        from trtutils.builder.onnx import extract_subgraph

        model = onnx.load(str(onnx_path))
        num_nodes = len(model.graph.node)
        with pytest.raises(ValueError, match="end_idx must be less"):
            extract_subgraph(model, 0, num_nodes + 100)

    @pytest.mark.xfail(
        reason="Source bug: onnx_graphsurgeon Variable objects are not hashable in newer versions",
        strict=False,
    )
    def test_accepts_graph_surgeon_graph(self, onnx_path):
        """Works with gs.Graph input (skips gs.import_onnx call)."""
        import onnx
        import onnx_graphsurgeon as gs

        from trtutils.builder.onnx import extract_subgraph

        model = onnx.load(str(onnx_path))
        graph = gs.import_onnx(model)
        graph.cleanup().toposort()

        num_nodes = len(graph.nodes)
        if num_nodes < 2:
            pytest.skip("Model has too few nodes")

        subgraph = extract_subgraph(graph, 0, num_nodes - 1)
        assert isinstance(subgraph, onnx.ModelProto)

    @pytest.mark.xfail(
        reason="Source bug: onnx_graphsurgeon Variable objects are not hashable in newer versions",
        strict=False,
    )
    def test_extract_first_nodes(self, onnx_path):
        """Extract from index 0."""
        import onnx

        from trtutils.builder.onnx import extract_subgraph

        model = onnx.load(str(onnx_path))
        num_nodes = len(model.graph.node)
        if num_nodes < 2:
            pytest.skip("Model has too few nodes")

        subgraph = extract_subgraph(model, 0, 0)
        assert isinstance(subgraph, onnx.ModelProto)

    @pytest.mark.xfail(
        reason="Source bug: onnx_graphsurgeon Variable objects are not hashable in newer versions",
        strict=False,
    )
    def test_extract_last_nodes(self, onnx_path):
        """Extract ending at last node."""
        import onnx

        from trtutils.builder.onnx import extract_subgraph

        model = onnx.load(str(onnx_path))
        num_nodes = len(model.graph.node)
        if num_nodes < 2:
            pytest.skip("Model has too few nodes")

        subgraph = extract_subgraph(model, num_nodes - 1, num_nodes - 1)
        assert isinstance(subgraph, onnx.ModelProto)


@pytest.mark.cpu
class TestSplitModel:
    """Tests for split_model()."""

    @pytest.mark.xfail(
        reason="Source bug: onnx_graphsurgeon Variable objects are not hashable in newer versions",
        strict=False,
    )
    def test_split_basic(self, onnx_path):
        """Split a model at a valid index."""
        import onnx

        from trtutils.builder.onnx import split_model

        model = onnx.load(str(onnx_path))
        num_nodes = len(model.graph.node)
        if num_nodes < 3:
            pytest.skip("Model has too few nodes for split test")

        mid = num_nodes // 2
        subgraphs = split_model(model, [mid])
        assert isinstance(subgraphs, list)
        assert len(subgraphs) == 2

    @pytest.mark.xfail(
        reason="Source bug: onnx_graphsurgeon Variable objects are not hashable in newer versions",
        strict=False,
    )
    def test_split_into_three(self, onnx_path):
        """Split with two split points produces 3 subgraphs."""
        import onnx

        from trtutils.builder.onnx import split_model

        model = onnx.load(str(onnx_path))
        num_nodes = len(model.graph.node)
        if num_nodes < 5:
            pytest.skip("Model has too few nodes for triple split")

        third = num_nodes // 3
        two_thirds = 2 * num_nodes // 3
        subgraphs = split_model(model, [third, two_thirds])
        assert len(subgraphs) == 3
        for sg in subgraphs:
            assert isinstance(sg, onnx.ModelProto)

    def test_split_empty_indices(self, onnx_path):
        """ValueError for empty split indices."""
        import onnx

        from trtutils.builder.onnx import split_model

        model = onnx.load(str(onnx_path))
        with pytest.raises(ValueError, match="non-empty list"):
            split_model(model, [])

    def test_split_negative_index(self, onnx_path):
        """ValueError for negative split index."""
        import onnx

        from trtutils.builder.onnx import split_model

        model = onnx.load(str(onnx_path))
        with pytest.raises(ValueError, match="greater than 0"):
            split_model(model, [-1])

    def test_split_index_too_large(self, onnx_path):
        """ValueError when split index >= num_nodes."""
        import onnx

        from trtutils.builder.onnx import split_model

        model = onnx.load(str(onnx_path))
        num_nodes = len(model.graph.node)
        with pytest.raises(ValueError, match="less than the number of nodes"):
            split_model(model, [num_nodes + 100])

    @pytest.mark.xfail(
        reason="Source bug: onnx_graphsurgeon Variable objects are not hashable in newer versions",
        strict=False,
    )
    def test_subgraphs_are_valid_onnx(self, onnx_path):
        """Each split part is a valid ONNX model."""
        import onnx

        from trtutils.builder.onnx import split_model

        model = onnx.load(str(onnx_path))
        num_nodes = len(model.graph.node)
        if num_nodes < 3:
            pytest.skip("Model has too few nodes")

        mid = num_nodes // 2
        subgraphs = split_model(model, [mid])
        for sg in subgraphs:
            assert isinstance(sg, onnx.ModelProto)
            assert len(sg.graph.node) > 0


@pytest.mark.cpu
class TestExtractSubgraphFromFile:
    """Tests for file-based extraction."""

    @pytest.mark.xfail(
        reason="Source bug: onnx_graphsurgeon Variable objects are not hashable in newer versions",
        strict=False,
    )
    def test_extract_from_file(self, onnx_path, tmp_path):
        """extract_subgraph_from_file creates output file."""
        import onnx

        from trtutils.builder.onnx import extract_subgraph_from_file

        model = onnx.load(str(onnx_path))
        num_nodes = len(model.graph.node)
        if num_nodes < 2:
            pytest.skip("Model has too few nodes")

        output = tmp_path / "sub.onnx"
        extract_subgraph_from_file(onnx_path, output, 0, num_nodes - 1)
        assert output.exists()

    @pytest.mark.xfail(
        reason="Source bug: onnx_graphsurgeon Variable objects are not hashable in newer versions",
        strict=False,
    )
    def test_split_from_file(self, onnx_path, tmp_path):
        """split_model_from_file creates output files."""
        import onnx

        from trtutils.builder.onnx import split_model_from_file

        model = onnx.load(str(onnx_path))
        num_nodes = len(model.graph.node)
        if num_nodes < 3:
            pytest.skip("Model has too few nodes")

        out_dir = tmp_path / "splits"
        out_dir.mkdir()
        mid = num_nodes // 2
        split_model_from_file(onnx_path, out_dir, [mid])
        assert any(out_dir.iterdir())
