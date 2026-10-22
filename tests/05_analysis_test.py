from pathlib import Path
import shutil
import numpy as np
import torch
import vame
import pytest
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from unittest.mock import patch
from vame.util.gif_pose_helper import background
from vame.util.model_util import load_model, load_training_metadata
from vame.util.auxiliary import read_states
from vame.io.load_poses import read_pose_estimation_file
from vame.model.dataloader import load_normalization
from vame.preprocessing.to_model import format_xarray_for_rnn
from vame.analysis.pose_segmentation import embed_latent_vectors, sliding_windows
from vame.visualization import visualize_umap, generate_reports, visualize_hierarchical_tree


def _copy_trained_project(setup_project_and_train_model, tmp_path, **config_updates):
    """Copy the shared trained project so a test can segment it without changing it"""
    config = setup_project_and_train_model["config_data"]
    project_path = tmp_path / Path(config["project_path"]).name
    shutil.copytree(config["project_path"], project_path, symlinks=True)
    return {**config, "project_path": str(project_path), **config_updates}


@pytest.mark.parametrize("segmentation_algorithm", ["hmm", "kmeans"])
def test_pose_segmentation_files_exists(setup_project_and_train_model, tmp_path, segmentation_algorithm):
    config = _copy_trained_project(
        setup_project_and_train_model,
        tmp_path,
        segmentation_algorithms=[segmentation_algorithm],
        hmm_trained=False,
    )
    with patch("builtins.input", return_value="yes"):
        vame.segment_session(config=config, overwrite_segmentation=True, save_logs=True)
    file = config["session_names"][0]
    n_clusters = config["n_clusters"]
    save_base_path = Path(config["project_path"]) / "results" / file / config["model_name"]
    latent_vector_path = save_base_path / "latent_vectors.npy"
    motif_usage_path = save_base_path / f"{segmentation_algorithm}-{n_clusters}" / f"motif_usage_{file}.npy"

    assert latent_vector_path.exists()
    assert motif_usage_path.exists()
    assert read_states(config)["segment_session"]["execution_state"] == "success"


def test_pose_segmentation_pretrained_hmm(setup_project_and_train_model, tmp_path):
    """hmm_trained=True reuses the saved HMM and gives the same labels"""
    config = _copy_trained_project(setup_project_and_train_model, tmp_path, segmentation_algorithms=["hmm"])
    session = config["session_names"][0]
    n_clusters = config["n_clusters"]
    label_path = (
        Path(config["project_path"]) / "results" / session / config["model_name"]
        / f"hmm-{n_clusters}" / f"{n_clusters}_hmm_label_{session}.npy"
    )
    hmm_path = Path(config["project_path"]) / "results" / "hmm_trained.pkl"
    hmm_path.unlink(missing_ok=True)

    vame.segment_session(config={**config, "hmm_trained": False}, overwrite_segmentation=True)
    assert hmm_path.exists()
    fitted_labels = np.load(label_path)
    label_path.unlink()

    vame.segment_session(config={**config, "hmm_trained": True}, overwrite_segmentation=True)
    np.testing.assert_array_equal(np.load(label_path), fitted_labels)


def test_motif_videos_mp4_files_exists(setup_project_and_segment):
    vame.motif_videos(
        config=setup_project_and_segment["config_data"],
        save_logs=True,
    )
    project_path = setup_project_and_segment["config_data"]["project_path"]
    file = setup_project_and_segment["config_data"]["session_names"][0]
    model_name = setup_project_and_segment["config_data"]["model_name"]
    n_clusters = setup_project_and_segment["config_data"]["n_clusters"]
    segmentation_algorithms = ["hmm", "kmeans"]
    for seg in segmentation_algorithms:
        save_base_path = Path(project_path) / "results" / file / model_name / f"{seg}-{n_clusters}" / "cluster_videos"
        assert len(list(save_base_path.glob("*.mp4"))) > 0
        assert len(list(save_base_path.glob("*.mp4"))) <= n_clusters
        # Clips are always H.264 MP4, whatever format the source video was in.
        assert not list(save_base_path.glob("*.avi"))


def test_cohort_community_files_exists(setup_project_and_segment):
    vame.community(
        config=setup_project_and_segment["config_data"],
        cut_tree=2,
        save_logs=True,
    )
    project_path = setup_project_and_segment["config_data"]["project_path"]
    n_clusters = setup_project_and_segment["config_data"]["n_clusters"]
    segmentation_algorithms = ["hmm", "kmeans"]
    for seg in segmentation_algorithms:
        base_path = Path(project_path) / "results" / "community_cohort" / f"{seg}-{n_clusters}"
        cohort_path = base_path / "cohort_transition_matrix.npy"
        community_path = base_path / "cohort_community_label.npy"
        cohort_segmentation_algorithm_path = base_path / f"cohort_{seg}_label.npy"
        cohort_community_bag_path = base_path / "cohort_community_bag.npy"

        assert cohort_path.exists()
        assert community_path.exists()
        assert cohort_segmentation_algorithm_path.exists()
        assert cohort_community_bag_path.exists()


def test_community_videos_mp4_files_exists(setup_project_and_segment):
    vame.community_videos(
        config=setup_project_and_segment["config_data"],
        save_logs=True,
    )
    file = setup_project_and_segment["config_data"]["session_names"][0]
    model_name = setup_project_and_segment["config_data"]["model_name"]
    n_clusters = setup_project_and_segment["config_data"]["n_clusters"]
    project_path = setup_project_and_segment["config_data"]["project_path"]
    segmentation_algorithms = ["hmm", "kmeans"]
    for seg in segmentation_algorithms:
        save_base_path = (
            Path(project_path) / "results" / file / model_name / f"{seg}-{n_clusters}" / "community_videos"
        )
        assert len(list(save_base_path.glob("*.mp4"))) > 0
        assert len(list(save_base_path.glob("*.mp4"))) <= n_clusters
        # Clips are always H.264 MP4, whatever format the source video was in.
        assert not list(save_base_path.glob("*.avi"))


def test_visualization_output_files(setup_project_and_segment):
    visualize_umap(
        config=setup_project_and_segment["config_data"],
        save_to_file=True,
        save_logs=True,
    )
    project_path = setup_project_and_segment["config_data"]["project_path"]
    session_names = setup_project_and_segment["config_data"]["session_names"]
    images_base_path = Path(project_path) / "reports" / "umap"
    for ses in session_names:
        assert len(list(images_base_path.glob("umap_*.png"))) > 0


@pytest.mark.parametrize(
    "mode,segmentation_algorithm",
    [
        ("sampling", "hmm"),
        ("reconstruction", "hmm"),
        ("sampling", "kmeans"),
        ("reconstruction", "kmeans"),
        ("centers", "kmeans"),
    ],
)
def test_generative_model_figures(
    setup_project_and_segment,
    mode,
    segmentation_algorithm,
):
    generative_figure = vame.generative_model(
        config=setup_project_and_segment["config_data"],
        segmentation_algorithm=segmentation_algorithm,
        mode=mode,
        save_logs=True,
    )
    assert isinstance(generative_figure, Figure)


def test_report(setup_project_and_segment):
    generate_reports(config=setup_project_and_segment["config_data"], show_figure=False)
    reports_path = Path(setup_project_and_segment["config_data"]["project_path"]) / "reports"
    assert len(list(reports_path.glob("*.png"))) > 0


def test_generative_kmeans_wrong_mode(setup_project_and_segment):
    with pytest.raises(ValueError):
        vame.generative_model(
            config=setup_project_and_segment["config_data"],
            segmentation_algorithm="hmm",
            mode="centers",
            save_logs=True,
        )


# @pytest.mark.parametrize("label", [None, "community", "motif"])
# def test_gif_frames_files_exists(setup_project_and_evaluate_model, label):
#     with patch("builtins.input", return_value="yes"):
#         vame.segment_session(setup_project_and_evaluate_model["config_data"])

#     def mock_background(
#         project_path=None,
#         session=None,
#         video_path=None,
#         num_frames=None,
#         save_background=True,
#     ):
#         num_frames = 100
#         return background(
#             project_path=project_path,
#             session=session,
#             video_path=video_path,
#             num_frames=num_frames,
#             save_background=save_background,
#         )

#     SEGMENTATION_ALGORITHM = "hmm"
#     VIDEO_LEN = 30
#     vame.community(
#         config=setup_project_and_evaluate_model["config_data"],
#         cut_tree=2,
#         save_logs=False,
#     )
#     visualize_umap(
#         config=setup_project_and_evaluate_model["config_data"],
#         segmentation_algorithm=SEGMENTATION_ALGORITHM,
#         label=label,
#         save_logs=False,
#     )
#     with patch("vame.util.gif_pose_helper.background", side_effect=mock_background):
#         vame.gif(
#             config_path=setup_project_and_evaluate_model["config_path"],
#             segmentation_algorithm=SEGMENTATION_ALGORITHM,
#             pose_ref_index=[0, 5],
#             subtract_background=True,
#             start=None,
#             length=VIDEO_LEN,
#             max_lag=30,
#             label=label,
#             file_format=".mp4",
#             crop_size=(300, 300),
#         )

#     video = setup_project_and_evaluate_model["config_data"]["session_names"][0]
#     model_name = setup_project_and_evaluate_model["config_data"]["model_name"]
#     n_clusters = setup_project_and_evaluate_model["config_data"]["n_clusters"]

#     save_base_path = (
#         Path(setup_project_and_evaluate_model["config_data"]["project_path"])
#         / "results"
#         / video
#         / model_name
#         / f"{SEGMENTATION_ALGORITHM}-{n_clusters}"
#     )

#     gif_frames_path = save_base_path / "gif_frames"
#     assert len(list(gif_frames_path.glob("*.png"))) == VIDEO_LEN


def test_embedding_reads_training_variable(setup_project_and_train_model):
    """Segmentation encodes the same preprocessed variable the model was trained on"""
    config = setup_project_and_train_model["config_data"]
    training_variable = load_training_metadata(config)["parameters"]["read_from_variable"]
    assert training_variable == "position_scaled"

    with patch(
        "vame.analysis.pose_segmentation.format_xarray_for_rnn",
        wraps=format_xarray_for_rnn,
    ) as mock_format:
        embed_latent_vectors(
            config=config,
            sessions=config["session_names"][:1],
            fixed=config["egocentric_data"],
            overwrite=True,
        )
    assert mock_format.call_args.kwargs["read_from_variable"] == training_variable


def test_embedding_matches_normalized_encoding(setup_project_and_train_model):
    """Segmentation latents equal encoding a window normalized as in training"""
    config = setup_project_and_train_model["config_data"]
    session = config["session_names"][0]
    latent_vectors = embed_latent_vectors(
        config=config,
        sessions=[session],
        fixed=config["egocentric_data"],
        overwrite=True,
    )[0]

    parameters = load_training_metadata(config)["parameters"]
    file_path = Path(config["project_path"]) / "data" / "processed" / f"{session}_processed.nc"
    _, _, ds = read_pose_estimation_file(file_path=str(file_path))
    data, _ = format_xarray_for_rnn(
        ds=ds,
        read_from_variable=parameters["read_from_variable"],
        keypoints=parameters["keypoints_used"],
        extra_features=parameters.get("extra_features"),
    )
    mean, std = load_normalization(Path(config["project_path"]) / "data" / "train")
    i = 10
    window = (data[:, i : i + config["time_window"]].T - mean) / std

    model = load_model(config, config["model_name"], config["egocentric_data"])
    device = next(model.parameters()).device
    with torch.no_grad():
        mu, _, _ = model.lmbda(model.encoder(torch.from_numpy(window[None]).float().to(device)))
    np.testing.assert_allclose(latent_vectors[i], mu.cpu().numpy()[0], atol=1e-4)


def test_embedding_session_shorter_than_time_window_raises(setup_project_and_train_model):
    """A session with fewer frames than time_window raises"""
    config = {**setup_project_and_train_model["config_data"], "time_window": 10**7}
    with pytest.raises(ValueError, match="fewer than time_window"):
        embed_latent_vectors(
            config=config,
            sessions=config["session_names"][:1],
            fixed=config["egocentric_data"],
            overwrite=True,
        )


def test_sliding_windows_layout():
    """Window i holds frames i to i + temp_win - 1 as (time, features), also when n_features == temp_win"""
    temp_win = n_features = 4
    data = np.arange(n_features * 10, dtype=float).reshape(n_features, 10)
    windows = sliding_windows(data, temp_win)
    assert windows.shape == (10 - temp_win + 1, temp_win, n_features)
    for i in range(len(windows)):
        np.testing.assert_array_equal(windows[i], data[:, i : i + temp_win].T)


def test_segment_session_failure_raises_and_records_failed(tmp_path):
    """A failing segmentation raises and is recorded as failed in states.json"""
    (tmp_path / "states").mkdir()
    (tmp_path / "states" / "states.json").write_text("{}")
    config = {"project_path": str(tmp_path)}
    with pytest.raises(KeyError):
        vame.segment_session(config, save_logs=False)
    assert read_states(config)["segment_session"]["execution_state"] == "failed"


def test_visualize_hierarchical_tree_without_showing(tmp_path):
    """show_figure=False returns the tree figure without calling plt.show"""
    tree_dir = tmp_path / "results" / "community_cohort" / "hmm-2"
    tree_dir.mkdir(parents=True)
    plt.imsave(tree_dir / "tree.png", np.zeros((4, 4)))
    config = {"project_path": str(tmp_path), "n_clusters": 2}
    with patch("vame.visualization.community.plt.show") as mock_show:
        fig = visualize_hierarchical_tree(config, segmentation_algorithm="hmm", show_figure=False)
    mock_show.assert_not_called()
    assert isinstance(fig, Figure)
    plt.close(fig)


@pytest.fixture
def video_config(tmp_path):
    (tmp_path / "states").mkdir()
    (tmp_path / "states" / "states.json").write_text("{}")
    for session in ("s1", "s2"):
        (tmp_path / "results" / session / "VAME" / "hmm-2").mkdir(parents=True)
    return {
        "project_path": str(tmp_path),
        "session_names": ["s1", "s2"],
        "all_data": "no",
        "model_name": "VAME",
        "n_clusters": 2,
        "segmentation_algorithms": ["hmm"],
    }


@pytest.mark.parametrize("video_function", [vame.motif_videos, vame.community_videos])
def test_videos_for_given_sessions(video_config, video_function):
    """Passing sessions writes videos for those sessions only, without prompting"""
    with (
        patch("vame.analysis.videowriter.create_cluster_videos") as mock_create,
        patch("builtins.input") as mock_input,
    ):
        video_function(config=video_config, sessions=["s2"], save_logs=False)
    mock_input.assert_not_called()
    assert [c.kwargs["session"] for c in mock_create.call_args_list] == ["s2"]


@pytest.mark.parametrize("video_function", [vame.motif_videos, vame.community_videos])
def test_videos_unknown_session_raises(video_config, video_function):
    with patch("vame.analysis.videowriter.create_cluster_videos") as mock_create:
        with pytest.raises(ValueError, match="Unknown session"):
            video_function(config=video_config, sessions=["s3"], save_logs=False)
    mock_create.assert_not_called()
