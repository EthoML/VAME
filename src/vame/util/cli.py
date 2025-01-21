from typing import List


def get_sessions_from_user_input(
    cfg: dict,
    action_message: str = "run this step",
) -> List[str]:
    user_input = input(
        f"Do you want to {action_message} for your entire dataset? \n"
        "If you only want to use a specific session, type the session name \n"
        "yes/no/<session_name>: "
    )
    if user_input in ["Yes", "yes"]:
        sessions = cfg["session_names"]
    elif user_input in ["No", "no"]:
        sessions = []
        for session in cfg["session_names"]:
            use_session = input("Do you want to use " + session + "? yes/no: ")
            if use_session in ["Yes", "yes"]:
                sessions.append(session)
            else:
                continue
    else:
        if user_input in cfg["session_names"]:
            sessions = [user_input]
        else:
            raise ValueError("Invalid input. Please enter yes, no, or a valid session name.")
