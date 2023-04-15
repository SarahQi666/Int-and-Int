
import copy as cp
import numpy as np

from pyskl.smp import warning_r0
from ..builder import PIPELINES


@PIPELINES.register_module()
class TimeWindowSampleFrames_J:
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







    def _jnt_next_sv(self,num_frames,num_keypoint,num_person,real_num_person,kp):
        """Define the intensity(sv) of human movement,which decides the position of the clip.

        Args:
            num_frames (int): The number of frames.（“t总”）
            num_keypoint (int): The number of keypoints.（“N”）
            num_person (int): The number of person(max).(即call的num_persons)
            real_num_person (list(int)): The number of people in every frame.(即call的num_persons[t],名得改)
        """

        sv=[] 


        for t in range(0,num_frames):
            if t==num_frames-1:
                sv.append(0)
            else:
                xx=[]
                sv_t=kp[:,t+1,:,:] - kp[:,t,:,:]

                for n in range(0,num_person): 
                    for i in range(0,num_keypoint):
                        x=np.linalg.norm(sv_t[n,i,:]/100.,ord=None)


                        if (x!=np.inf) and (x>=0) and (np.isnan(x)==False):
                            xx.append(x)
                sv_t=sum(xx)/10.

                sv_t=sv_t/max(1,real_num_person[t],real_num_person[t+1])
                sv.append(sv_t)

        return sv
    

    def _jnt_avr_sv(self,num_frames,num_keypoint,num_person,real_num_person,kp):
        """Define the intensity(sv) of human movement,which decides the position of the clip.

        Args:
            num_frames (int): The number of frames.（“t总”）
            num_keypoint (int): The number of keypoints.（“N”）
            num_person (int): The number of person(max).(即call的num_persons)
            real_num_person (list(int)): The number of people in every frame.(即call的num_persons[t],名得改)
        """
        sv=[] 
        

        kp_bar=np.zeros((num_person,num_keypoint,2))
        for n in range(num_person):
            for i in range(num_keypoint):
                kp_bar[n,i,:]=np.mean(kp[n,:,i,:],axis=0)


        for t in range(num_frames):
            xx=[]
            sv_t=kp[:,t,:,:] - kp_bar[:,:,:]
            for n in range(num_person):
                for i in range(num_keypoint):

                    x=np.linalg.norm(sv_t[n,i,:]/100.,ord=None)
                    if (x!=np.inf) and (x>=0) and (np.isnan(x)==False):
                        xx.append(x)
            sv_t=sum(xx)/10.
            sv_t=sv_t/max(1,real_num_person[t])
            sv.append(sv_t)



        return sv
    

    def _boneC_next_sv(self,num_frames,num_keypoint,num_person,real_num_person,kp):
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
        


        bone_Center=[]
        for a_key in bone_pairs:
            the_joint = a_key

            vec=(0.5*kp[:,:,11,:2]+0.5*kp[:,:,12,:2]) - kp[:,:,the_joint,:2] 
            bone_Center.append(vec) 
        bone=np.array(bone_Center).transpose(1,2,0,3)
        

        for t in range(num_frames):
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
        

        for t in range(num_frames):
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
    

    def _boneC_avr_sv(self,num_frames,num_keypoint,num_person,real_num_person,kp):
        """Define the intensity(sv) of human movement,which decides the position of the clip.

        Args:
            num_frames (int): The number of frames.（“t总”）
            num_keypoint (int): The number of keypoints.（“N”）
            real_num_person (list(int)): The number of people in every frame.
        """
        sv=[] 
        

        kp_bar=np.zeros((num_person,num_keypoint,2))
        for n in range(num_person):
            for i in range(num_keypoint):
                kp_bar[n,i,:]=np.mean(kp[n,:,i,:],axis=0)


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
        


        bone_Center=[]
        bone_bar_Center=[]
        for a_key in bone_pairs:
            the_joint = a_key

            vec=(0.5*kp[:,:,11,:2]+0.5*kp[:,:,12,:2]) - kp[:,:,the_joint,:2] 
            bone_Center.append(vec) 

            vec_bar=(0.5*kp_bar[:,11,:2]+0.5*kp_bar[:,12,:2]) - kp_bar[:,the_joint,:2] 
            bone_bar_Center.append(vec_bar)
        
        bone=np.array(bone_Center).transpose(1,2,0,3)

        bone_bar=np.array(bone_bar_Center).transpose(1,0,2)



        for t in range(num_frames):
            xx=[]
            for n in range(num_person):
                if not np.all(np.abs(bone[n,t]) < 1e-5): 
                    for i in range(num_keypoint):
                        sv_t=10*(1.0 - (np.dot(bone_bar[n,i,:], bone[n,t,i,:])/(np.linalg.norm(bone_bar[n,i,:]) * np.linalg.norm(bone[n,t,i,:]))))

                        if (sv_t!=np.inf) and (sv_t>=0) and (np.isnan(sv_t)==False):
                            xx.append(sv_t) 
            sv_t=sum(xx)
            sv_t=sv_t/max(1,real_num_person[t])
            sv.append(sv_t)

        return sv
    
    def _boneL_avr_sv(self,num_frames,num_keypoint,num_person,real_num_person,kp):
        """Define the intensity(sv) of human movement,which decides the position of the clip.

        Args:
            num_frames (int): The number of frames.（“t总”）
            num_keypoint (int): The number of keypoints.（“N”）
            real_num_person (list(int)): The number of people in every frame.
        """
        sv=[] 
        

        kp_bar=np.zeros((num_person,num_keypoint,2))
        for n in range(num_person):
            for i in range(num_keypoint):
                kp_bar[n,i,:]=np.mean(kp[n,:,i,:],axis=0)


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
        bone_bar_Local=[]
        for a_key in bone_pairs:
            a_bone_value = bone_pairs[a_key]
            the_joint = a_key
            v1 = a_bone_value

            vec=kp[:, :, v1, :2] - kp[:,:,the_joint,:2] 
            bone_Local.append(vec) 

            vec_bar=kp_bar[:, v1, :2] - kp_bar[:,the_joint,:2] 
            bone_bar_Local.append(vec_bar)
        
        bone=np.array(bone_Local).transpose(1,2,0,3)
        bone_bar=np.array(bone_bar_Local).transpose(1,0,2)


        for t in range(num_frames):
            xx=[]
            for n in range(num_person):
                if not np.all(np.abs(bone[n,t]) < 1e-5): 
                    for i in range(num_keypoint):
                        sv_t=10*(1.0 - (np.dot(bone_bar[n,i,:], bone[n,t,i,:])/(np.linalg.norm(bone_bar[n,i,:]) * np.linalg.norm(bone[n,t,i,:]))))
                        if (sv_t!=np.inf) and (sv_t>=0) and (np.isnan(sv_t)==False):
                            xx.append(sv_t) 
            sv_t=sum(xx)
            sv_t=sv_t/max(1,real_num_person[t])
            sv.append(sv_t)

        return sv
    

    def _angC_next_sv(self,num_frames,num_keypoint,num_person,real_num_person,kp):
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
        


        angle_Center=[]
        tt=[]
        ang=[]
        for a_key in angle_pairs:
            the_joint = a_key

            veck=kp[:,:,the_joint,:2]-(0.5*kp[:,:,11,:2]+0.5*kp[:,:,12,:2]) 
            veckj=(0.5*kp[:,:,5,:2]+0.5*kp[:,:,6,:2])-(0.5*kp[:,:,11,:2]+0.5*kp[:,:,12,:2])
            for n in range(num_person):
                for t in range(num_frames):
                    ang_t = 1.0 - np.dot(veck[n,t,:], veckj[n,t,:])/(np.linalg.norm(veck[n,t,:]) * np.linalg.norm(veckj[n,t,:])) 
                    tt.append(ang_t)
                ang.append(tt)
            angle_Center.append(ang) 
        angle=np.array(angle_Center).transpose(1,2,0) 
        

        angle[np.isnan(angle)] = np.inf

        

        for t in range(num_frames):
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
    

    def _angC_avr_sv(self,num_frames,num_keypoint,num_person,real_num_person,kp):
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
        



        angle_Center=[]
        tt=[]
        ang=[]
        real_angle=[]
        for a_key in angle_pairs:
            the_joint = a_key

            veck=kp[:,:,the_joint,:2]-(0.5*kp[:,:,11,:2]+0.5*kp[:,:,12,:2]) 
            veckj=(0.5*kp[:,:,5,:2]+0.5*kp[:,:,6,:2])-(0.5*kp[:,:,11,:2]+0.5*kp[:,:,12,:2])
            for n in range(num_person):
                for t in range(num_frames):
                    ang_t = 1.0 - np.dot(veck[n,t,:], veckj[n,t,:])/(np.linalg.norm(veck[n,t,:]) * np.linalg.norm(veckj[n,t,:])) 
                    tt.append(ang_t)
                ang.append(tt)
            angle_Center.append(ang) 
        angle=np.array(angle_Center).transpose(1,2,0) 
        

        angle[np.isnan(angle)] = np.inf


        angle_bar=np.zeros((num_person,num_keypoint))
        for n in range(num_person):
            for t in range(num_frames):
                if not (np.all(np.abs(angle[n,t])==np.inf)):
                    real_angle.append(angle[n,t,:]) 
        for n in range(num_person):
            if len(np.array(real_angle).shape)==3:
                for i in range(num_keypoint):
                        angle_bar[n,i]=np.mean(np.array(real_angle)[n,:,i])
            elif len(np.array(real_angle).shape)==2:
                for i in range(num_keypoint):

                        angle_bar[n,i]=np.mean(np.array(real_angle)[:,i])


        for t in range(num_frames):
            yy=[]
            for n in range(num_person):
                if not (np.all(np.abs(angle[n,t])==np.inf)):
                    sv_t=np.abs(angle[n,t,:] - angle_bar[n,:])
                    sv_t=sv_t.sum()/10.

                    if (sv_t!=np.inf) and (sv_t>=0) and (np.isnan(sv_t)==False):
                        yy.append(sv_t)
            sv_t=sum(yy)
            sv_t=sv_t/max(1,real_num_person[t])
            sv.append(sv_t)

        return sv
        
    def _angL_avr_sv(self,num_frames,num_keypoint,num_person,real_num_person,kp):
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
        real_angle=[]
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


        angle_bar=np.zeros((num_person,num_keypoint))
        for n in range(num_person):
            for t in range(num_frames):
                if not (np.all(np.abs(angle[n,t])==np.inf)):
                    real_angle.append(angle[n,t,:]) 
        for n in range(num_person):
            if len(np.array(real_angle).shape)==3:
                for i in range(num_keypoint):
                        angle_bar[n,i]=np.mean(np.array(real_angle)[n,:,i])
            elif len(np.array(real_angle).shape)==2:
                for i in range(num_keypoint):

                        angle_bar[n,i]=np.mean(np.array(real_angle)[:,i])


        for t in range(num_frames):
            yy=[]
            for n in range(num_person):
                if not (np.all(np.abs(angle[n,t])==np.inf)):
                    sv_t=np.abs(angle[n,t,:] - angle_bar[n,:])
                    sv_t=sv_t.sum()/10.
                    if (sv_t!=np.inf) and (sv_t>=0) and (np.isnan(sv_t)==False):
                        yy.append(sv_t)
            sv_t=sum(yy)
            sv_t=sv_t/max(1,real_num_person[t])
            sv.append(sv_t)

        return sv

    



    def _get_train_clips(self, num_frames, clip_len,num_keypoint,num_person,real_num_person,kp):
        """Sample indices for motion clips.

        Args:
            num_frames (int): The number of frames.
            clip_len (int): The length of the clip.
        """
        allinds = []
        sv_jnt_next=self._jnt_next_sv(num_frames,num_keypoint,num_person,real_num_person,kp)
        sv_jnt_avr=self._jnt_avr_sv(num_frames,num_keypoint,num_person,real_num_person,kp)









        theta=[1,1,1,1,1,1,1,1,1,1]
        sv_J=list(np.add([(theta[0]/sum(theta)/max(float("1e-8"),max(sv_jnt_next)))*x for x in sv_jnt_next],[(theta[1]/sum(theta)/max(float("1e-8"),max(sv_jnt_avr)))*x for x in sv_jnt_avr]))







        sv=sv_J


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
        sv_jnt_next=self._jnt_next_sv(num_frames,num_keypoint,num_person,real_num_person,kp)
        sv_jnt_avr=self._jnt_avr_sv(num_frames,num_keypoint,num_person,real_num_person,kp)









        theta=[1,1,1,1,1,1,1,1,1,1]
        sv_J=list(np.add([(theta[0]/sum(theta)/max(float("1e-8"),max(sv_jnt_next)))*x for x in sv_jnt_next],[(theta[1]/sum(theta)/max(float("1e-8"),max(sv_jnt_avr)))*x for x in sv_jnt_avr]))







        sv=sv_J


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


