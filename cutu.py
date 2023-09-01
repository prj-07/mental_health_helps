import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from transformers import BertTokenizer, TFBertForMaskedLM
import re
from transformers import BertTokenizer, BertModel
from transformers import BertTokenizer
# import torch
from flask import Flask, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np
import os
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import BartTokenizer
import pandas as pd
json_data1 = '''
[
    {"title": "Talk therapy for depression may help lower heart disease risk", "date": "Published August 1, 2023", "content": "For people with depression, successful treatment with talk therapy may lower the risk for heart disease, perhaps by helping them adopt healthier habits, according to a 2023 study."},
    {"title": "Bonds that transcend age", "date": "Published August 1, 2023", "content": "Intergenerational friendships typically involve an older adult and someone who's 15, 20, or more years younger. Studies suggest people can benefit physically and psychologically from such friendships. Friends of diverse ages can expose people to different experiences, attitudes, and approaches. To create intergenerational friendships, people can use a shared workplace or activity as a springboard for deeper connection. They should show genuine interest in the other person's life and experiences."},
    {"title": "Denial: How it hurts, how it helps, and how to cope", "date": "Published July 26, 2023", "content": "Denial is a natural response at times when you're unable or unwilling to face the facts. As a defense mechanism, it can be helpful or harmful. Here's how to spot it in yourself and others, and how to move from denial toward meaningful change."},
    {"title": "What is somatic therapy?", "date": "Published July 7, 2023", "content": "Trauma can register within our bodies on a cellular level. What that means — and how best to heal from serious traumas — is the focus of somatic therapy, a newer form of mental health counseling that highlights how deeply painful experiences affect us and can be addressed through mind-body approaches."},
    {"title": "Lessons learned from COVID", "date": "Published July 1, 2023", "content": "While the COVID pandemic changed how many people approached their health, the experience offers a teachable moment about how people can maintain new and improved healthy habits. Four areas that were most affected by COVID and enabled people to create positive changes are exercise, diet, medical check-ups, and social connections."},
    {"title": "Regular physical activity can boost mood", "date": "Published July 1, 2023", "content": "A 2023 study suggests regular exercise improves symptoms of depression, anxiety, and psychological distress more effectively than sedentary behavior."},
    {"title": "How positive psychology can help you cultivate better heart health", "date": "Published June 1, 2023", "content": "Optimism and other positive emotions may guard against serious heart-related events and death. Simple exercises such as expressing gratitude and performing acts of kindness can counter depression and improve well-being. These interventions may encourage people—including those with heart-related conditions such as heart attacks and heart failure—to exercise more and take their medications more consistently. Having a more positive outlook may help reinforce other positive behaviors, or what psychologists refer to as the upward spiral. This momentum can help people start healthy habits like exercise, which then becomes self-reinforcing."},
    {"title": "Man's best (health) friend", "date": "Published June 1, 2023", "content": "Adopting a dog is one of the best ways for older adults to combat many common health issues. Research has shown that dog owners have lower risks of cardiovascular disease and take more daily steps compared with non-owners. Having a dog also can lower stress levels and help people become more social."},
    {"title": "Social challenges such as isolation linked to earlier death", "date": "Published June 1, 2023", "content": "A 2023 study suggests that certain social challenges, such as isolation, may be useful to help predict older adults' risk of earlier death."},
    {"title": "Silent suffering", "date": "Published June 1, 2023", "content": "Survivor's guilt includes strong, persistent feelings of sadness and remorse. Advances in cancer treatment have led to an unprecedented 18 million Americans who are cancer survivors, making survivor's guilt a possible psychological outcome for greater numbers. Survivor's guilt may be more common among people who have survived cancers with high death rates. Strategies to cope include taking time to grieve, seeking support from fellow survivors, and getting counseling to explore underlying contributors to feelings of guilt."}
]
'''


json_data2 = '''
[
    {
        "title": "What is mental health?",
        "content": "Mental health includes our emotional, psychological, and social well-being. It affects how we think, feel, and act as we cope with life. It also helps determine how we handle stress, relate to others, and make choices. Mental health is important at every stage of life, from childhood and adolescence through adulthood and aging."
    },
    {
        "title": "Why is mental health important?",
        "content": "Mental health is important because it can help you to:Cope with the stresses of lifeBe physically healthyHave good relationshipsMake meaningful contributions to your communityWork productivelyRealize your full potential"
    },
    {
        "title": "How can I improve my mental health?",
        "content": "There are many different things you can do to improve your mental health, including:Staying positive. It's important to try to have a positive outlook; some ways to do that include Finding balance between positive and negative emotions.  Staying positive doesn't mean that you never feel negative emotions, such as sadness or anger. You need to feel them so that you can move through difficult situations. They can help you to respond to a problem. But you don't want those emotions to take over. For example, it's not helpful to keep thinking about bad things that happened in the past or worry too much about the future. Trying to hold on to the positive emotions when you have themTaking a break from negative information. Know when to stop watching or reading the news. Use social media to reach out for support and feel connected to others but be careful. Don't fall for rumors, get into arguments, or negatively compare your life to others.Practicing gratitude, which means being thankful for the good things in your life. It's helpful to do this every day, either by thinking about what you are grateful for or writing it down in a journal. These can be big things, such as the support you have from loved ones, or little things, such as enjoying a nice meal. It's important to allow yourself a moment to enjoy that you had the positive experience.  Practicing gratitude can help you to see your life differently. For example, when you are stressed, you may not notice that there are also moments when you have some positive emotions. Gratitude can help you to recognize them.Taking care of your physical health, since your physical and mental health are connected. Some ways to take care of your physical health include Being physically active. Exercise can reduce feelings of stress and depression and improve your mood.Getting enough sleep. Sleep affects your mood. If you don't get a good sleep, you may become more easily annoyed and angry. Over the long term, a lack of quality sleep can make you more likely to become depressed. So it's important to make sure that you have a regular sleep schedule and get enough quality sleep every night.Healthy eating. Good nutrition will help you feel better physically but could also improve your mood and decrease anxiety and stress. Also, not having enough of certain nutrients may contribute to some mental illnesses. For example, there may be a link between low levels of vitamin B12 and depression. Eating a well-balanced diet can help you to get enough of the nutrients you need. Connecting with others. Humans are social creatures, and it's important to have strong, healthy relationships with others. Having good social support may help protect you against the harms of stress. It is also good to have different types of connections. Besides connecting with family and friends, you could find ways to get involved with your community or neighborhood. For example, you could volunteer for a local organization or join a group that is focused on a hobby you enjoy.Developing a sense of meaning and purpose in life. This could be through your job, volunteering, learning new skills, or exploring your spirituality.Developing coping skills, which are methods you use to deal with stressful situations. They may help you face a problem, take action, be flexible, and not easily give up in solving it.Meditation, which is a mind and body practice where you learn to focus your attention and awareness. There are many types, including mindfulness meditation and transcendental meditation. Meditation usually involvesA quiet location with as few distractions as possible A specific, comfortable posture. This could be sitting, lying down, walking, or another position. A focus of attention, such as a specially chosen word or set of words, an object, or your breathing An open attitude, where you try to let distractions come and go naturally without judging them Relaxation techniques are practices you do to produce your body's natural relaxation response. This slows down your breathing, lowers your blood pressure, and reduces muscle tension and stress. Types of relaxation techniques include Progressive relaxation, where you tighten and relax different muscle groups, sometimes while using mental imagery or breathing exercises Guided imagery, where you learn to focus on positive images in your mind, to help you feel more relaxed and focused Biofeedback, where you use electronic devices to learn to control certain body functions, such as breathing, heart rate, and muscle tensionSelf-hypnosis, where the goal is to get yourself into a relaxed, trance-like state when you hear a certain suggestion or see a specific cue.Deep breathing exercises, which involve focusing on taking slow, deep, even breathsIt's also important to recognize when you need to get help. Talk therapy and or medicines can treat mental disorders. If you don't know where to get treatment, start by contacting your primary care provider."
    },
    {
        "title": "My Mental Health: Do I Need Help?",
        "content": "Transforming the understanding and treatment of mental illnesses 8:30 a.m.  5 p.m. ET, M-F National Institute of Mental Health  Office of Science Policy, Planning, and Communications  6001 Executive Boulevard, Room 6200, MSC 9663Bethesda, MD 20892-966 The National Institute of Mental Health (NIMH) is part of the National Institutes of Health (NIH), a component of theU S. Department of Health and Human Services."
    },
    {
        "title": "Live Your Life Well",
        "content": "Some people think that only people with mental illnesses have to pay attention to their mental health. But the truth is that your emotions, thoughts and attitudes affect your energy, productivity and overall health. Good mental health strengthens your ability to cope with everyday hassles and more serious crises and challenges. Good mental health is essential to creating the life you want. Just as you brush your teeth or get a flu shot, you can take steps to promote your mental health. A great way to start is by learning to deal with stress. Stress can eat away at your well-being like acid eating away at your stomach. Actually, stress can contribute to stomach pains and lots of other problems, like: Stress also can lead to serious mental health problems, like . If you think you have such a problem, you can . Of course you can't magically zap all sources of stress. But you can learn to deal with them in a way that promotes the well-being you want--and deserve. Learn more about . The concrete steps we're suggesting are not based on guesses, fads or advice from grandma (though she probably got a lot right). They represent hundreds of research studies with thousands of participants, often conducted over decades and backed by major universities or government agencies. This research shows that how good you feel is to a fairly large extent up to you. No matter how stressful your situation, you can take steps to promote your well-being. We're not talking about huge changes to your lifestyle, either. We're talking about reasonable steps that if used consistently can increase your comfort and boost your ability to build a rewarding life. Mental Health America is the country's leading non-profit dedicated to promoting mental health. We have worked with communities, families, schools and individuals across the nation to ensure that all people have a chance to thrive. Founded 100 years ago to improve conditions for people with mental illnesses, we have worked tirelessly since then to promote understanding of anxiety disorders, depression and other mental health issues. Our more than 200 affiliate offices help veterans returning from war, victims of natural disasters, children at risk of substance abuse and millions of other people across the country. Now we are launching the Live Your Life Well campaign to provide tools to people like you who are stressed by the many demands of modern life. You also can get more information from your ."
    },
    {
        "title": "Caring for Your Mental Health",
        "content": "Transforming the understandingand treatment of mental illnesses.8:30 a.m. National Institute of Mental HealthOffice of Science Policy, Planning, and Communication Executive Boulevard, Room 6200, MSC 9663  Bethesda, MD 20892-9663The National Institute of Mental Health (NIMH) is part of the National Institutes of Health (NIH), a component of the U.S. Department of Health and Human Services."
    },
    {
        "title": "31 Tips to Boost Your Mental Health",
        "content": "1. Include 3 things you were grateful for and 3 things you were able to accomplish each day. Coffe consumption is linked to lower rates of depression. If you cant drink coffe because of the caffeine, try another good-for-you drink like green tea. 3. It could be camping with friends or a trip to the tropics. The act of planning a vacation and having something to look forward to can boost your overall happiness for up to 8 weeks! 4, Do something you're good at to build self-confidence, then tackle a tougher task 5.The optimal temperature for sleep is between 60 and 67 degrees Fahrenheit.- Martin Luther King, Jr. Think of something in your life you want to improve, and figure out what you can do to take a step in the right direction.  7.  with a new recipe, write a poem, paint or try a Pinterest project. Creative expression and overall well-being are linked.  8.  Close, quality, relationships are key for a happy, healthy life.  9.  The flavonoids, caffeine, and theobromine in chocolate are thought to work together to improve alertness and mental skills.  10.-Maya Angelou.If you have personal experience with mental illness or recovery, share on Twitter, Instagram and Tumblr with mentalillnessfeelslike. Check out what other people are saying  .  11.We just need to soak up the joy in the ones we've already got. Trying to be optimistic doesn't mean ignoring the uglier sides of life. It just means focusing on the positive as much as possible. 12.for about 20 minutes to help you clear your mind. Pick a design that's geometric and a little complicated for the best effect.Check out hundreds of free printable coloring pageang out with a funny friend, watch a comedy or check out cute videos online. Laughter helps reduce anxiety.  14.Leave your smart phone at home for a day and disconnect from constant emails, alerts, and other interruptions. Spend time doing something fun with someone face-to-face. 15.Not only will you get chores done, but dancing reduces levels of cortisol (the stress hormone), and increases endorphins (the body's feel-good chemicals).Studies suggest that yawning helps cool the brain and improves alertness and mental efficiency.  17. Try adding Epsom salts to soothe aches and pains and help boost magnesium levels, which can be depleted by stress.  18.Writing about upsetting experiences can reduce symptoms of depression. 19. Time with animals lowers the stress hormone - cortisol, and boosts oxytocin which stimulates feelings of happiness. If you dont have a pet, hang out with a friend who does or volunteer at a shelter.  20. Henry David Thoreau.Practice mindfulness by staying in the present. Try . 21.Often times people only explore attractions on trips, but you may be surprised what cool things are in your own backyard. 22.You'll save some time in the mornings and have a sense of control about the week ahead.  23.they are linked to decreased rates of depression and schizophrenia among their many benefits. Fish oil supplements work, but eating your omega-3s in foods like wild salmon, flaxseeds or walnuts also helps build healthy gut bacteria.  even if it's just forgiving that person who cut you off during your commute. People who forgive have better mental health and report being more satisfied with their lives.  25.Disraeli. Try to find the silver lining in something kind of cruddy that happened recently. It may not be the easiest thing to do, but smiling can help to lower your heart rate and calm you down.27. not for a material item, but to let someone know why you appreciate them. Written expressions of gratitude are linked to increased happiness.- have a cookout, go to a park, or play a game. People are 12 times more likely to feel happy on days that they spend 6-7 hours with friends and family.  29.- it could be a stroll through a park, or a hike in the woods. Research shows that being in nature can increase energy levels, reduce depression and boost well-being. 30 , and apply sunscreen. Sunlight synthesizes Vitamin D, which experts believe is a mood elevator.31. -Albert Einstein. Try something outside of your comfort zone to make room lor adventure and excitement in your life."}
]

'''
df1 = pd.read_json(json_data1, orient='records')
# print(df1)
df2 = pd.read_json(json_data2, orient='records')
# print(df2)

merged_df = pd.concat([df1, df2], ignore_index=True)

# print(merged_df)
df = merged_df
df = df.drop('date', axis=1)
print(df)

df['context'] = df['title']+df['content']


model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = TFBertForMaskedLM.from_pretrained(model_name)


# Load pre-trained BART model and tokenizer
# model_name = 'facebook/bart-large-cnn'

# # Now you can use BartTokenizer
# tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

# tokenizer = BartTokenizer.from_pretrained(model_name)

# model = BartForConditionalGeneration.from_pretrained(model_name)

# # Function to generate summaries using BART


# def generate_summary(text):
#     inputs = tokenizer.encode(
#         "summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
#     summary_ids = model.generate(
#         inputs, max_length=100, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
#     summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
#     return summary


# # Generate summaries and create a new 'summarised' column
# df['summarised'] = df['context'].apply(generate_summary)

# print(df.head())

df['summarised'] = df['context']
model_name = 'bert-base-uncased'

tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Function to get BERT embeddings for a sentence


# def get_bert_embedding(sentence):
#     inputs = tokenizer(sentence, return_tensors='pt',
#                        truncation=True, padding=True)
#     with torch.no_grad():
#         outputs = model(**inputs)
#         embeddings = outputs.last_hidden_state.mean(
#             dim=1)  # Average pooling over tokens

#     return embeddings


def get_bert_embedding(sentence):
    # Load the pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = TFBertModel.from_pretrained('bert-base-uncased')

    # Tokenize the input sentence
    inputs = tokenizer(sentence, return_tensors='tf',
                       truncation=True, padding=True)

    # Forward pass through the BERT model
    outputs = model(**inputs)

    # Extract the last hidden state and compute the mean
    embeddings = tf.reduce_mean(outputs.last_hidden_state, axis=1).numpy()

    return embeddings


# Add embeddings to DataFrame
df['embedding'] = df['summarised'].apply(get_bert_embedding)

print(df.head())


tokenizer.add_special_tokens({'pad_token': '[PAD]'})


# def get_bert_embedding(sentence):
#     inputs = tokenizer(sentence, return_tensors='pt',
#                        truncation=True, padding=True)
#     with torch.no_grad():
#         outputs = model(**inputs)
#         embeddings = outputs.last_hidden_state.mean(
#             dim=1)  # Average pooling over tokens

#     return embeddings


app = Flask(__name__)


# def get_bert_embedding(sentence):
#     inputs = tokenizer(sentence, return_tensors='pt',
#                        truncation=True, padding=True)
#     with torch.no_grad():
#         outputs = model(**inputs)
#         # print(outputs)

#         # embeddings = outputs.hidden_states.mean(dim=1)  # Adjust the attribute name
#         embeddings = outputs.last_hidden_state.mean(
#             dim=1)  # Using the second-to-last hidden state

#     return embeddings


print(df['embedding'][0])
print(type(df['embedding'][0]))


# def parse_embedding_string(embedding_string):
#     # Extract numerical values from the string using regular expression
#     values = re.findall(r'-?\d+\.\d+', embedding_string)
#     # Convert the values to floats and construct a tensor
#     embedding_tensor = torch.tensor([float(value) for value in values])
#     return embedding_tensor


# # Apply the parsing function to each entry in the 'embedding' column
# df['embedding'] = df['embedding'].apply(parse_embedding_string)

# Verify tensor shapes
# for tensor in df['embedding']:
#     if isinstance(tensor, torch.Tensor):
#         print(tensor.shape)

df_embeddings = np.vstack(df['embedding'])
# Verify the shape of df_embeddings
print(df_embeddings.shape)

df_embeddings_array = np.array(df_embeddings)
print(df_embeddings_array)


@app.route('/process_query45', methods=['POST'])
def process_query45():
    data = request.json
    print(data)
    query = data.get("query", "")
    print(query)
    print(1)
    # return jsonify(data)

    # def get_embedding(text, model="text-embedding-ada-002"):
    # text = text.replace("\n", " ")
    # return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']

    query_embedding = get_bert_embedding(query)
    print(2)
    # print(query_embedding)
    print(31)
    df_embeddings = np.vstack(df['embedding'])
    # print(df_embeddings)
    print(df_embeddings)
    print(32)
    # import numpy as np


# Calculate cosine similarities

    df_embeddings_array = np.array(df_embeddings)
    print(41)
    print(df_embeddings_array)
    query_embedding_array = query_embedding
    print(42)
    print(query_embedding_array)
    print(43)
    similarities = cosine_similarity(
        df_embeddings, query_embedding.reshape(1, -1))

# Print the similarities array for inspection
    print(5)
    print(similarities)

    def find_most_similar(query_embedding):
        query_embedding = np.array(query_embedding)
        similarities = cosine_similarity(
            np.vstack(df['embedding']), query_embedding.reshape(1, -1))
        most_similar_index = similarities.argmax()
        return most_similar_index

    most_similar_index = find_most_similar(query_embedding)
    print(most_similar_index)
    most_similar_row = df.iloc[most_similar_index]

    article_context = most_similar_row['context']

    article_content = most_similar_row['content']
    # response = openai.Completion.create(
    #     engine='text-davinci-003',
    #     prompt=article_content,
    #     max_tokens=100,
    #     temperature=0.7
    # )
    print(3)
#     input_ids = tokenizer.encode(article_content, return_tensors="pt")
#     output = model.generate(input_ids, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2, temperature=0.0)

# # Decode the generated output
#     print(4)

#     generated_response = tokenizer.decode(output[0], skip_special_tokens=True)
    # generated_response = response.choices[0].text.strip()
    print(5)
    generated_response = article_content
    return jsonify({'query': query, 'response': generated_response})


if __name__ == '__main__':
    app.run()
