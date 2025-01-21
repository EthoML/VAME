# from pynwb import NWBHDF5IO
# from pynwb.file import NWBFile
# from hdmf.utils import LabelledDict
# import pandas as pd


# def get_pose_data_from_nwb_file(
#     nwbfile: NWBFile,
#     path_to_pose_nwb_series_data: str,
# ) -> LabelledDict:
#     """
#     Get pose data from nwb file using a inside path to the nwb data.

#     Parameters
#     ---------
#     nwbfile : NWBFile)
#         NWB file object.
#     path_to_pose_nwb_series_data : str
#         Path to the pose data inside the nwb file.

#     Returns
#     -------
#     LabelledDict
#         Pose data.
#     """
#     if not path_to_pose_nwb_series_data:
#         raise ValueError("Path to pose nwb series data is required.")
#     pose_data = nwbfile
#     for key in path_to_pose_nwb_series_data.split("/"):
#         if isinstance(pose_data, dict):
#             pose_data = pose_data.get(key)
#             continue
#         pose_data = getattr(pose_data, key)
#     return pose_data


# def get_dataframe_from_pose_nwb_file(
#     file_path: str,
#     path_to_pose_nwb_series_data: str,
# ) -> pd.DataFrame:
#     """
#     Get pose data from nwb file and return it as a pandas DataFrame.

#     Parameters
#     ---------
#     file_path : str
#         Path to the nwb file.
#     path_to_pose_nwb_series_data : str
#         Path to the pose data inside the nwb file.

#     Returns
#     -------
#     pd.DataFrame
#         Pose data as a pandas DataFrame.
#     """
#     with NWBHDF5IO(file_path, "r") as io:
#         nwbfile = io.read()
#         pose = get_pose_data_from_nwb_file(nwbfile, path_to_pose_nwb_series_data)
#         dataframes = []
#         for label, pose_series in pose.items():
#             data = pose_series.data[:]
#             confidence = pose_series.confidence[:]
#             df = pd.DataFrame(data, columns=[f"{label}_x", f"{label}_y"])
#             df[f"likelihood_{label}"] = confidence
#             dataframes.append(df)
#         final_df = pd.concat(dataframes, axis=1)
#     return final_df
