
import copy as cp
import numpy as np

from pyskl.smp import warning_r0
from ..builder import PIPELINES


@PIPELINES.register_module()
class TimeWindowSampleFrames_angL_next:
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






    def _angL_next_sv(self,num_frames,num_keypoint,num_person,real_num_person,kp):
        """Define the intensity(sv) of human movement,which decides the position of the clip.

        Args:
            num_frames (int): The number of frames.（“t总”）
            num_keypoint (int): The number of keypoints.（“N”）
            real_num_person (list(int)): The number of people in every frame.
        """
        sv=[] 


        angle_pairs = {
            9: (7,5), 
            7: (9,5),
            5: (7,6), 
            6: (5,8), 
            8: (6,10),
            10: (8,6),
            3: (1,0),
            1: (3,0),
            2: (0,4),
            4: (2,0),
            0: (5,11), 
            15: (13,11),
            13: (15,11),
            11: (5,13),
            12: (6,14),
            14: (12,16),
            16: (14,12)
        }
        


        angle_Local=[]
        tt=[]
        ang=[]
        for a_key in angle_pairs:
            a_angle_value = angle_pairs[a_key]
            the_joint = a_key

            v1 = a_angle_value[0]
            v2 = a_angle_value[1]
            vec1=kp[:,:,v1,:2]-kp[:,:,the_joint,:2] 
            vec2=kp[:,:,v2,:2]-kp[:,:,the_joint,:2]
            for n in range(num_person):
                for t in range(num_frames):
                    ang_t = 1.0 - np.dot(vec1[n,t,:], vec2[n,t,:])/(np.linalg.norm(vec1[n,t,:]) * np.linalg.norm(vec2[n,t,:])) 
                    tt.append(ang_t)
                ang.append(tt)
            angle_Local.append(ang) 
        angle=np.array(angle_Local).transpose(1,2,0) 
        

        angle[np.isnan(angle)] = np.inf
        

        for t in range(0,num_frames):
            if t==num_frames-1:
                sv.append(0)
            else:
                yy=[]
                for n in range(num_person):
                    if not (np.all(np.abs(angle[n,t])==np.inf) or np.all(np.abs(angle[n,t+1])==np.inf)):
                        sv_t=np.abs(angle[n,t+1,:] - angle[n,t,:])
                        sv_t=sv_t.sum()

                        if (sv_t!=np.inf) and (sv_t>=0) and (np.isnan(sv_t)==False):
                            yy.append(sv_t)
                sv_t=sum(yy)
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
        sv=self._angL_next_sv(num_frames,num_keypoint,num_person,real_num_person,kp)
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
        sv=self._angL_next_sv(num_frames,num_keypoint,num_person,real_num_person,kp)
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


