
import copy as cp
import numpy as np

from pyskl.smp import warning_r0
from ..builder import PIPELINES


@PIPELINES.register_module()
class TimeWindowSampleFrames_boneL_next:
    """TimeWindow sample frames from the video.

    To sample an n-frame clip from the video.To make the testing results reproducible, a
    random seed is set during testing, to make the sampling results
    deterministic.

    Required keys are "total_frames", "start_index" , added or modified keys
    are "frame_inds", "clip_len", "frame_interval" and "num_clips".

    Args:
        clip_len (int): Frames of each sampled output clip. 是n？
        num_clips (int): Number of clips to be sampled. Default: 1.
        seed (int): The random seed used during test time. Default: 255.
    """

    def __init__(self,
                 clip_len,
                 num_clips=1,
                 p_interval=1,
                 seed=255,
                 **deprecated_kwargs):

        self.clip_len = clip_len
        self.num_clips = num_clips
        self.seed = seed
        self.p_interval = p_interval
        if not isinstance(p_interval, tuple):
            self.p_interval = (p_interval, p_interval)
        if len(deprecated_kwargs): 
            warning_r0('[TimeWindowSampleFrames] The following args has been deprecated: ')
            for k, v in deprecated_kwargs.items():
                warning_r0(f'Arg name: {k}; Arg value: {v}')






    def _boneL_next_sv(self,num_frames,num_keypoint,num_person,real_num_person,kp):
        """Define the intensity(sv) of human movement,which decides the position of the clip.

        Args:
            num_frames (int): The number of frames.（“t总”）
            num_keypoint (int): The number of keypoints.（“N”）
            real_num_person (list(int)): The number of people in every frame.
        """
        sv=[] 


        bone_pairs = {
            9: 7,
            7: 5,
            5: 0,
            10: 8,
            8: 6,
            6: 0,
            0: 9,
            3: 1,
            1: 0,
            4: 2,
            2: 0,
            15: 13,
            13: 11,
            11: 5,
            16: 14,
            14: 12,
            12: 6
        }
        


        bone_Local=[]
        for a_key in bone_pairs:
            a_bone_value = bone_pairs[a_key]
            the_joint = a_key
            v1 = a_bone_value

            vec=kp[:, :, v1, :2] - kp[:,:,the_joint,:2] 
            bone_Local.append(vec) 
        bone=np.array(bone_Local).transpose(1,2,0,3)
        

        for t in range(0,num_frames):
            if t==num_frames-1:
                sv.append(0)
            else:
                xx=[]
                for n in range(num_person):
                    if not (np.all(np.abs(bone[n,t]) < 1e-5) or np.all(np.abs(bone[n,t+1]) < 1e-5)):
                        for i in range(num_keypoint):
                            sv_t=10*(1.0 - (np.dot(bone[n,t+1,i,:], bone[n,t,i,:])/(np.linalg.norm(bone[n,t+1,i,:]) * np.linalg.norm(bone[n,t,i,:]))))

                            if (sv_t!=np.inf) and (sv_t>=0) and (np.isnan(sv_t)==False):
                                xx.append(sv_t)
                sv_t=sum(xx)
                sv_t=sv_t/max(1,min(real_num_person[t],real_num_person[t+1]))
                sv.append(sv_t)



        return sv

    




























    
    



    def _get_train_clips(self, num_frames, clip_len,num_keypoint,num_person,real_num_person,kp):
        """Sample indices for motion clips.

        Args:
            num_frames (int): The number of frames.
            clip_len (int): The length of the clip.
        """
        allinds = []
        sv=self._boneL_next_sv(num_frames,num_keypoint,num_person,real_num_person,kp)
        for clip_idx in range(self.num_clips):

            svmax_idx=sv.index(max(sv))



            if num_frames < clip_len:
                inds = np.arange(0, clip_len)
            elif num_frames >= clip_len:
                if clip_len//2<=svmax_idx<=clip_len//2+(num_frames-clip_len):
                    start_idx=svmax_idx-clip_len//2
                    inds = np.arange(start_idx, start_idx+clip_len)
                elif svmax_idx<clip_len//2:
                    inds = np.arange(0,clip_len)
                elif svmax_idx>clip_len//2+(num_frames-clip_len):
                    inds = np.arange(num_frames-clip_len,num_frames)

            allinds.append(inds)


        return np.concatenate(allinds)
    
    def _get_test_clips(self, num_frames, clip_len,num_keypoint,num_person,real_num_person,kp):
        """Sample indices for motion clips.

        Args:
            num_frames (int): The number of frames.
            clip_len (int): The length of the clip.
        """
        np.random.seed(self.seed) 
        
        allinds = []
        sv=self._boneL_next_sv(num_frames,num_keypoint,num_person,real_num_person,kp)
        for clip_idx in range(self.num_clips):

            svmax_idx=sv.index(max(sv))



            if num_frames < clip_len:
                inds = np.arange(0, clip_len)
            elif num_frames >= clip_len:
                if clip_len//2<=svmax_idx<=clip_len//2+(num_frames-clip_len):
                    start_idx=svmax_idx-clip_len//2
                    inds = np.arange(start_idx, start_idx+clip_len)
                elif svmax_idx<clip_len//2:
                    inds = np.arange(0,clip_len)
                elif svmax_idx>clip_len//2+(num_frames-clip_len):
                    inds = np.arange(num_frames-clip_len,num_frames)

            allinds.append(inds)


        return np.concatenate(allinds)
    


    def __call__(self, results):
        num_frames = results['total_frames']
        kp = results['keypoint']
        assert num_frames == kp.shape[1]
        num_person = kp.shape[0]
        num_keypoint = kp.shape[2]

        real_num_person = [num_person] * num_frames 
        for i in range(num_frames):
            j = num_person - 1
            while j >= 0 and np.all(np.abs(kp[j, i]) < 1e-5):
                j -= 1
            real_num_person[i] = j + 1 

        transitional = [False] * num_frames 

        for i in range(1, num_frames - 1):
            if real_num_person[i] != real_num_person[i - 1]:
                transitional[i] = transitional[i - 1] = True
            if real_num_person[i] != real_num_person[i + 1]:
                transitional[i] = transitional[i + 1] = True

        if results.get('test_mode', False):
            inds = self._get_test_clips(num_frames, self.clip_len,num_keypoint,num_person,real_num_person,kp)
        else:
            inds = self._get_train_clips(num_frames, self.clip_len,num_keypoint,num_person,real_num_person,kp)

        inds = np.mod(inds, num_frames) 
        start_index = results['start_index']
        inds = inds + start_index
 
        inds_int = inds.astype(np.int) 
        coeff = np.array([transitional[i] for i in inds_int]) 
        inds = (coeff * inds_int + (1 - coeff) * inds).astype(np.float32) 

        results['frame_inds'] = inds.astype(np.int)


        results['clip_len'] = self.clip_len
        results['frame_interval'] = None
        results['num_clips'] = self.num_clips
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'num_clips={self.num_clips}, '
                    f'seed={self.seed})')
        return repr_str


