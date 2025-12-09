import base64
from io import BytesIO

from animalai import AnimalAIEnvironment
from mlagents_envs.base_env import DecisionSteps
from PIL import Image
import numpy as np

from src.definitions.prompts.prompts import OBSERVATIONS
from src.definitions.cardinal_directions import action_name_to_action_tuple
from src.vision.vision import AAIVisualObservation, VisionSystem


class CameraSystem(VisionSystem):
    def __init__(self):
        super().__init__()

    @property
    def observation_prompt(self) -> str:
        return OBSERVATIONS["paper"]



    def get_observation(self,
                        env: AnimalAIEnvironment,
                        save: bool = True,
                        save_path: str = "observation.jpg",
                        show: bool = False,
                        border_width: int = 2,
                        ) -> AAIVisualObservation:
        behavior = list(env.behavior_specs.keys())[0]  # by default should be AnimalAI?team=0
        dec, _ = env.get_steps(behavior)
        camera_obs = env.get_obs_dict(dec.obs)["camera"]

        # Verify observation format (expected: channels-first RGB)
        assert len(camera_obs.shape) == 3 and camera_obs.shape[0] == 3, (
            f"Expected camera observation in channels-first format (3, H, W), "
            f"but got shape {camera_obs.shape}. This may indicate an incompatible "
            f"AnimalAI version (Expected version 5.x.x)"
        )

        # Convert from channels-first (C, H, W) to channels-last (H, W, C)
        camera_obs = np.transpose(camera_obs, (1, 2, 0))
        scaled_image_array = np.array(camera_obs * 255, dtype=np.uint8)
        assert scaled_image_array.all() < 256
        image = Image.fromarray(scaled_image_array)

        if save: image.save(save_path)
        if show: image.show()
        base64_string = self._convert_image_to_base64_string(image)
        return "", base64_string

    @staticmethod
    def _convert_image_to_base64_string(image: Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        base64_encoded_data = base64.b64encode(buffered.getvalue())
        base64_string = base64_encoded_data.decode('utf-8')
        return base64_string
