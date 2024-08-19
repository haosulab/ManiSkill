import itertools
from typing import Dict, List, Optional, Tuple

import numpy as np
import sapien
import torch
from sapien import physx

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.registration import register_agent
from mani_skill.agents.robots.allegro_hand.allegro import AllegroHandRight
from mani_skill.utils import sapien_utils
from mani_skill.utils.structs.actor import Actor


@register_agent()
class AllegroHandRightTouch(AllegroHandRight):
    uid = "allegro_hand_right_touch"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/allegro/variation/allegro_hand_right_fsr_simple.urdf"

    def __init__(self, *args, **kwargs):
        # Order: thumb finger, index finger, middle finger, ring finger, from finger root to fingertip
        self.finger_fsr_link_names = [
            # allegro thumb has a different hardware design compared with other fingers
            "link_14.0_fsr",
            "link_15.0_fsr",
            "link_15.0_tip_fsr",
            # the hardware design of index, middle and ring finger are the same
            "link_1.0_fsr",
            "link_2.0_fsr",
            "link_3.0_tip_fsr",
            "link_5.0_fsr",
            "link_6.0_fsr",
            "link_7.0_tip_fsr",
            "link_9.0_fsr",
            "link_10.0_fsr",
            "link_11.0_tip_fsr",
        ]
        self.palm_fsr_link_names = [
            "link_base_fsr",
            "link_0.0_fsr",
            "link_4.0_fsr",
            "link_8.0_fsr",
        ]

        super().__init__(*args, **kwargs)

        self.pair_query: Dict[
            str, Tuple[physx.PhysxGpuContactPairImpulseQuery, Tuple[int, int, int]]
        ] = dict()
        self.body_query: Optional[
            Tuple[physx.PhysxGpuContactBodyImpulseQuery, Tuple[int, int, int]]
        ] = None

    def _after_init(self):
        super()._after_init()
        self.fsr_links: List[Actor] = sapien_utils.get_objs_by_names(
            self.robot.get_links(),
            self.palm_fsr_link_names + self.finger_fsr_link_names,
        )

    def get_fsr_obj_impulse(self, obj: Actor = None):
        if physx.is_gpu_enabled():
            px: physx.PhysxGpuSystem = self.scene.px
            # Create contact query if it is not existed
            if obj.name not in self.pair_query:
                bodies = list(zip(*[link._bodies for link in self.fsr_links]))
                bodies = list(itertools.chain(*bodies))
                obj_bodies = [
                    elem for item in obj._bodies for elem in itertools.repeat(item, 2)
                ]
                body_pairs = list(zip(bodies, obj_bodies))
                query = px.gpu_create_contact_pair_impulse_query(body_pairs)
                self.pair_query[obj.name] = (
                    query,
                    (len(obj._bodies), len(self.fsr_links), 3),
                )

            # Query contact buffer
            query, contacts_shape = self.pair_query[obj.name]
            px.gpu_query_contact_pair_impulses(query)
            contacts = (
                query.cuda_impulses.torch()
                .clone()
                .reshape((len(self.fsr_links), *contacts_shape))
            )  # [n, 16, 3]

            return contacts

        else:
            internal_fsr_links = [link._bodies[0].entity for link in self.fsr_links]
            contacts = self.scene.get_contacts()
            obj_contacts = sapien_utils.get_multiple_pairwise_contacts(
                contacts, obj._bodies[0].entity, internal_fsr_links
            )
            sorted_contacts = [obj_contacts[link] for link in internal_fsr_links]
            contact_forces = [
                sapien_utils.compute_total_impulse(contact)
                for contact in sorted_contacts
            ]

            return np.stack(contact_forces)

    def get_fsr_impulse(self):
        if physx.is_gpu_enabled():
            px: physx.PhysxGpuSystem = self.scene.px
            # Create contact query if it is not existed
            if self.body_query is None:
                # Convert the order of links so that the link from the same sub-scene will come together
                # It makes life easier for reshape
                bodies = list(zip(*[link._bodies for link in self.fsr_links]))
                bodies = list(itertools.chain(*bodies))

                query = px.gpu_create_contact_body_impulse_query(bodies)
                self.body_query = (
                    query,
                    (len(self.fsr_links[0]._bodies), len(self.fsr_links), 3),
                )

            # Query contact buffer
            query, contacts_shape = self.body_query
            px.gpu_query_contact_body_impulses(query)
            contacts = (
                query.cuda_impulses.torch().clone().reshape(*contacts_shape)
            )  # [n, 16, 3]

            return contacts

        else:
            internal_fsr_links = [link._bodies[0].entity for link in self.fsr_links]
            contacts = self.scene.get_contacts()
            contact_map = sapien_utils.get_cpu_actors_contacts(
                contacts, internal_fsr_links
            )
            sorted_contacts = [contact_map[link] for link in internal_fsr_links]
            contact_forces = [
                sapien_utils.compute_total_impulse(contact)
                for contact in sorted_contacts
            ]

            contact_impulse = torch.from_numpy(
                np.stack(contact_forces)[None, ...]
            )  # [1, 16, 3]
            return contact_impulse

    def get_proprioception(self):
        """
        Get the proprioceptive state of the agent.
        """
        obs = super().get_proprioception()
        fsr_impulse = self.get_fsr_impulse()
        obs.update({"fsr_impulse": torch.linalg.norm(fsr_impulse, dim=-1)})

        return obs
