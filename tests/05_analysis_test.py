from pathlib import Path
import vame
import pytest
from matplotlib.figure import Figure
from unittest.mock import patch
from vame.util.gif_pose_helper import background
from vame.visualization import visualize_umap


@pytest.mark.parametrize(
    "individual_segmentation,segmentation_algorithm,hmm_trained",
    [
        (True, "hmm", False),
        (False, "hmm", False),
        (False, "hmm", True),
        (True, "kmeans", False),
        (False, "kmeans", False),
    ],
)
def test_pose_segmentation_hmm_files_exists(
    setup_project_and_train_model,
    individual_segmentation,
    segmentation_algorithm,
    hmm_trained,
):
    mock_config = {
        **setup_project_and_train_model["config_data"],
        "individual_segmentation": individual_segmentation,
    }
    mock_config["hmm_trained"] = hmm_trained
    # with patch("vame.analysis.pose_segmentation.read_config", return_value=mock_config) as mock_read_config:
    with patch("builtins.input", return_value="yes"):
        vame.segment_session(
            config=setup_project_and_train_model["config_data"],
            save_logs=True,
        )
    project_path = setup_project_and_train_model["config_data"]["project_path"]
    file = setup_project_and_train_model["config_data"]["session_names"][0]
    model_name = setup_project_and_train_model["config_data"]["model_name"]
    n_clusters = setup_project_and_train_model["config_data"]["n_clusters"]
    save_base_path = Path(project_path) / "results" / file / model_name / f"{segmentation_algorithm}-{n_clusters}"
    latent_vector_path = save_base_path / f"latent_vector_{file}.npy"
    motif_usage_path = save_base_path / f"motif_usage_{file}.npy"

    assert latent_vector_path.exists()
    assert motif_usage_path.exists()


def test_motif_videos_mp4_files_exists(setup_project_and_train_model):
    vame.motif_videos(
        config=setup_project_and_train_model["config_data"],
        output_video_type=".mp4",
        save_logs=True,
    )
    project_path = setup_project_and_train_model["config_data"]["project_path"]
    file = setup_project_and_train_model["config_data"]["session_names"][0]
    model_name = setup_project_and_train_model["config_data"]["model_name"]
    n_clusters = setup_project_and_train_model["config_data"]["n_clusters"]
    segmentation_algorithms = ["hmm", "kmeans"]
    for seg in segmentation_algorithms:
        save_base_path = Path(project_path) / "results" / file / model_name / f"{seg}-{n_clusters}" / "cluster_videos"
        assert len(list(save_base_path.glob("*.mp4"))) > 0
        assert len(list(save_base_path.glob("*.mp4"))) <= n_clusters


def test_motif_videos_avi_files_exists(setup_project_and_train_model):
    # Check if the files are created
    vame.motif_videos(
        config=setup_project_and_train_model["config_data"],
        output_video_type=".avi",
        save_logs=True,
    )
    project_path = setup_project_and_train_model["config_data"]["project_path"]
    file = setup_project_and_train_model["config_data"]["session_names"][0]
    model_name = setup_project_and_train_model["config_data"]["model_name"]
    n_clusters = setup_project_and_train_model["config_data"]["n_clusters"]
    segmentation_algorithms = ["hmm", "kmeans"]
    for seg in segmentation_algorithms:
        save_base_path = Path(project_path) / "results" / file / model_name / f"{seg}-{n_clusters}" / "cluster_videos"
        assert len(list(save_base_path.glob("*.avi"))) > 0
        assert len(list(save_base_path.glob("*.avi"))) <= n_clusters


def test_cohort_community_files_exists(setup_project_and_train_model):
    vame.community(
        config=setup_project_and_train_model["config_data"],
        cut_tree=2,
        save_logs=True,
    )
    project_path = setup_project_and_train_model["config_data"]["project_path"]
    n_clusters = setup_project_and_train_model["config_data"]["n_clusters"]
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


def test_community_videos_mp4_files_exists(setup_project_and_train_model):
    vame.community_videos(
        config=setup_project_and_train_model["config_data"],
        output_video_type=".mp4",
        save_logs=True,
    )
    file = setup_project_and_train_model["config_data"]["session_names"][0]
    model_name = setup_project_and_train_model["config_data"]["model_name"]
    n_clusters = setup_project_and_train_model["config_data"]["n_clusters"]
    project_path = setup_project_and_train_model["config_data"]["project_path"]
    segmentation_algorithms = ["hmm", "kmeans"]
    for seg in segmentation_algorithms:
        save_base_path = (
            Path(project_path) / "results" / file / model_name / f"{seg}-{n_clusters}" / "community_videos"
        )
        assert len(list(save_base_path.glob("*.mp4"))) > 0
        assert len(list(save_base_path.glob("*.mp4"))) <= n_clusters


def test_community_videos_avi_files_exists(setup_project_and_train_model):
    vame.community_videos(
        config=setup_project_and_train_model["config_data"],
        output_video_type=".avi",
        save_logs=True,
    )
    file = setup_project_and_train_model["config_data"]["session_names"][0]
    model_name = setup_project_and_train_model["config_data"]["model_name"]
    n_clusters = setup_project_and_train_model["config_data"]["n_clusters"]
    project_path = setup_project_and_train_model["config_data"]["project_path"]
    segmentation_algorithms = ["hmm", "kmeans"]
    for seg in segmentation_algorithms:
        save_base_path = (
            Path(project_path) / "results" / file / model_name / f"{seg}-{n_clusters}" / "community_videos"
        )
        assert len(list(save_base_path.glob("*.avi"))) > 0
        assert len(list(save_base_path.glob("*.avi"))) <= n_clusters


@pytest.mark.parametrize(
    "label,segmentation_algorithm",
    [
        (None, "hmm"),
        ("motif", "hmm"),
        ("community", "hmm"),
        (None, "kmeans"),
        ("motif", "kmeans"),
        ("community", "kmeans"),
    ],
)
def test_visualization_output_files(
    setup_project_and_train_model,
    label,
    segmentation_algorithm,
):
    visualize_umap(
        config=setup_project_and_train_model["config_data"],
        segmentation_algorithm=segmentation_algorithm,
        label=label,
        save_logs=True,
    )

    project_path = setup_project_and_train_model["config_data"]["project_path"]
    file = setup_project_and_train_model["config_data"]["session_names"][0]
    model_name = setup_project_and_train_model["config_data"]["model_name"]
    n_clusters = setup_project_and_train_model["config_data"]["n_clusters"]

    project_path = setup_project_and_train_model["config_data"]["project_path"]

    save_base_path = (
        Path(project_path) / "results" / file / model_name / f"{segmentation_algorithm}-{n_clusters}" / "community"
    )
    assert len(list(save_base_path.glob(f"umap_vis*{file}.png"))) > 0


@pytest.mark.parametrize(
    "mode,segmentation_algorithm",
    [
        ("sampling", "hmm"),
        ("reconstruction", "hmm"),
        ("motifs", "hmm"),
        ("sampling", "kmeans"),
        ("reconstruction", "kmeans"),
        ("motifs", "kmeans"),
        ("centers", "kmeans"),
    ],
)
def test_generative_model_figures(
    setup_project_and_train_model,
    mode,
    segmentation_algorithm,
):
    generative_figure = vame.generative_model(
        config=setup_project_and_train_model["config_data"],
        segmentation_algorithm=segmentation_algorithm,
        mode=mode,
        save_logs=True,
    )
    assert isinstance(generative_figure, Figure)


@pytest.mark.parametrize(
    "segmentation_algorithm",
    ["hmm", "kmeans"],
)
def test_report(
    setup_project_and_train_model,
    segmentation_algorithm,
):
    vame.report(
        config=setup_project_and_train_model["config_data"],
        segmentation_algorithm=segmentation_algorithm,
    )
    reports_path = Path(setup_project_and_train_model["config_data"]["project_path"]) / "reports"
    assert len(list(reports_path.glob("*.png"))) > 0


def test_generative_kmeans_wrong_mode(setup_project_and_train_model):
    with pytest.raises(ValueError):
        vame.generative_model(
            config=setup_project_and_train_model["config_data"],
            segmentation_algorithm="hmm",
            mode="centers",
            save_logs=True,
        )


@pytest.mark.parametrize("label", [None, "community", "motif"])
def test_gif_frames_files_exists(setup_project_and_evaluate_model, label):
    with patch("builtins.input", return_value="yes"):
        vame.segment_session(setup_project_and_evaluate_model["config_data"])

    def mock_background(
        project_path=None,
        session=None,
        video_path=None,
        num_frames=None,
        save_background=True,
    ):
        num_frames = 100
        return background(
            project_path=project_path,
            session=session,
            video_path=video_path,
            num_frames=num_frames,
            save_background=save_background,
        )

    SEGMENTATION_ALGORITHM = "hmm"
    VIDEO_LEN = 30
    vame.community(
        config=setup_project_and_evaluate_model["config_data"],
        cut_tree=2,
        save_logs=False,
    )
    visualize_umap(
        config=setup_project_and_evaluate_model["config_data"],
        segmentation_algorithm=SEGMENTATION_ALGORITHM,
        label=label,
        save_logs=False,
    )
    with patch("vame.util.gif_pose_helper.background", side_effect=mock_background):
        vame.gif(
            config_path=setup_project_and_evaluate_model["config_path"],
            segmentation_algorithm=SEGMENTATION_ALGORITHM,
            pose_ref_index=[0, 5],
            subtract_background=True,
            start=None,
            length=VIDEO_LEN,
            max_lag=30,
            label=label,
            file_format=".mp4",
            crop_size=(300, 300),
        )

    video = setup_project_and_evaluate_model["config_data"]["session_names"][0]
    model_name = setup_project_and_evaluate_model["config_data"]["model_name"]
    n_clusters = setup_project_and_evaluate_model["config_data"]["n_clusters"]

    save_base_path = (
        Path(setup_project_and_evaluate_model["config_data"]["project_path"])
        / "results"
        / video
        / model_name
        / f"{SEGMENTATION_ALGORITHM}-{n_clusters}"
    )

    gif_frames_path = save_base_path / "gif_frames"
    assert len(list(gif_frames_path.glob("*.png"))) == VIDEO_LEN
