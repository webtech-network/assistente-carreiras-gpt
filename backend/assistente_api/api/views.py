import os
import sys
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(base_dir)

from lang_chain_cursos import chat
import json
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response

# Create your views here.
class BaseView(APIView):
    def post(self, request):
        body_unicode = request.body.decode('utf-8')
        body_obj = json.loads(body_unicode)
        
        chat_history = body_obj["history"]
        question  = body_obj["userInput"]

       
        interaction = chat(question, chat_history)
        return Response(interaction)
        