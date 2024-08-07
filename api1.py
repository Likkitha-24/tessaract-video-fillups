from flask import Flask, request, jsonify
import boto3
import json
import time
import os
from botocore.exceptions import NoCredentialsError, ClientError
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import Bedrock
import requests

app = Flask(__name__)

# AWS Credentials


# Initialize AWS clients
s3_client = boto3.client('s3', 
                         aws_access_key_id=aws_access_key_id,
                         aws_secret_access_key=aws_secret_access_key,
                         region_name=region_name)

transcribe_client = boto3.client('transcribe',
                                 aws_access_key_id=aws_access_key_id,
                                 aws_secret_access_key=aws_secret_access_key,
                                 region_name=region_name)

bedrock_client = boto3.client('bedrock-runtime',
                              aws_access_key_id=aws_access_key_id,
                              aws_secret_access_key=aws_secret_access_key,
                              region_name=region_name)

BUCKET_NAME = 'video-to-text-generate--1'

# Function to create S3 bucket if it doesn't exist
def create_bucket(bucket_name):
    try:
        s3_client.head_bucket(Bucket=bucket_name)
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            try:
                s3_client.create_bucket(Bucket=bucket_name)
            except ClientError as e:
                return {"error": f"Failed to create bucket: {e}"}, 500
        else:
            return {"error": f"Error checking bucket: {e}"}, 500

create_bucket(BUCKET_NAME)

@app.route('/process-video', methods=['POST'])
def process_video():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    try:
        # Upload file to S3
        filename = f"video_{int(time.time())}.{file.filename.split('.')[-1]}"
        s3_client.upload_fileobj(file, BUCKET_NAME, filename)
        
        # Start transcription job
        transcribe_job_name = f"transcribe_{int(time.time())}"
        transcribe_client.start_transcription_job(
            TranscriptionJobName=transcribe_job_name,
            Media={'MediaFileUri': f's3://{BUCKET_NAME}/{filename}'},
            MediaFormat=filename.split('.')[-1],
            LanguageCode='en-US'
        )

        # Wait for transcription job to complete
        start_time = time.time()
        while True:
            status = transcribe_client.get_transcription_job(TranscriptionJobName=transcribe_job_name)
            if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
                break
            if time.time() - start_time > 300:  # 5 minutes timeout
                return jsonify({"error": "Transcription job timed out"}), 504
            time.sleep(5)

        if status['TranscriptionJob']['TranscriptionJobStatus'] == 'COMPLETED':
            transcription_url = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
            response = requests.get(transcription_url)
            transcript_text = json.loads(response.text)['results']['transcripts'][0]['transcript']

            # Generate True/False questions
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_text(transcript_text)

            prompt_template = PromptTemplate(
                input_variables=["text"],
                template="""Generate True/False questions from the following text. 
                Each question should be on a new line and should have an answer (True/False).
                Text: {text}
                Questions with Answers:"""
            )

            llm = Bedrock(
                model_id="anthropic.claude-v2", 
                client=bedrock_client,
                model_kwargs={"max_tokens_to_sample": 500}
            )
            chain = LLMChain(llm=llm, prompt=prompt_template)

            all_questions = []
            for text in texts:
                if isinstance(text, str):
                    result = chain.run(text)
                    all_questions.append(result)

            return jsonify({
                "transcript": transcript_text,
                "questions": all_questions
            }), 200
        else:
            return jsonify({"error": "Transcription job failed"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)