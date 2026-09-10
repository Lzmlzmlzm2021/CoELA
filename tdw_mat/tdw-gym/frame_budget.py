def frame_budget_exhausted(current_frames, frames_this_step, max_frames):
    """Return whether another TDW frame would cross the episode boundary."""
    return current_frames + frames_this_step >= max_frames
