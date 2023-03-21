import whisper, os, sys
from moviepy.editor import *
# from gensim import
import gensim
import openai
import tempfile

class Transcribe:

    def __init__(self, video_path):
        self.video_path = video_path

    def extract_audio(self):
        video = VideoFileClip(self.video_path).subclip(44,2640)
        temp_dir = tempfile.TemporaryDirectory()
        audio_path = os.path.join(temp_dir.name,"Audio.wav")               
        video.audio.write_audiofile(audio_path)
        return audio_path,temp_dir

    def generate_transcription(self, audio_path, temp_dir):
        model = whisper.load_model("base")
        # transcription = model.transcribe(input_path)
        transcription = model.transcribe(audio_path)
        output_path = os.path.join(temp_dir.name,"Transcript.txt")
        with open(output_path,"w") as f:
            f.write(transcription["text"])
        return output_path
    #Doesn't Work
    def get_summary():

        with open("./Audio3.txt","r") as f:
            trans_file = f.read()
        tokens = list(gensim.utils.tokenize(trans_file))
        tokens_list = [tokens[itr:itr+1500] for itr in range(0,len(tokens),1500)]
        openai.api_key=""
        engine_list = openai.Engine.list()
        for sub_token_list in tokens_list:
            response = openai.Completion.create(engine="davinci",prompt="Summarize the following - "+" ".join(sub_token_list),temperature=0.3,
                                                max_tokens=140,
                                                top_p=1,
                                                frequency_penalty=0,
                                                presence_penalty=0,
                                                stop=["\n"]
                                                )
            with open("./Summary3.txt", "a") as f:
                f.write(response["choices"][0]["text"])

        return


if __name__=="__main__":
    video_path = sys.argv[1]
    # Transcribe.get_summary()
    transcribe = Transcribe(video_path)
    audio_path, temp_dir = transcribe.extract_audio()
    output_path = transcribe.generate_transcription(audio_path, temp_dir)
    print(output_path)
