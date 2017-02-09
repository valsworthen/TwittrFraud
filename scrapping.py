from twython import Twython
import pandas as pd
import time

consumer_key = "wDUW3FM2jREw6b3jh0qqn5KCy"
consumer_secret = "LINDjFbul5pCrsOKuM5pnEVZq2ZTKCL70emmG6fZMxlUdkmcfA"
access_token = "816672911050293248-j1QickwIvsWtiwJ3KoI0e5jB2EPTneH"
access_secret = "V7HUQzBTZnvjYVENmS4rP0NYRiG0kulCMWdjiFcdEocYH"

twitter = Twython(consumer_key, consumer_secret,
                  access_token, access_secret)

#geocode = '48.856720,2.343596,100km'


def flatten(d):
    for key in list(d):
        if isinstance(d[key], list):
            value = d.pop(key)
            for i, v in enumerate(value):
                d.update(flatten({'%s__%s' % (key, i): v}))
        elif isinstance(d[key], dict):
            value = d.pop(key)
            d.update([('%s__%s' % (key, sub), v) for (sub, v) in flatten(value).items()])
    return d

    
dict_tweet = {'contributors': [],
 'coordinates': [],
 'created_at': [],
 'entities__media__0__display_url': [],
 'entities__media__0__expanded_url': [],
 'entities__media__0__id': [],
 'entities__media__0__id_str': [],
 'entities__media__0__indices__0': [],
 'entities__media__0__indices__1': [],
 'entities__media__0__media_url': [],
 'entities__media__0__media_url_https': [],
 'entities__media__0__sizes__large__h': [],
 'entities__media__0__sizes__large__resize': [],
 'entities__media__0__sizes__large__w': [],
 'entities__media__0__sizes__medium__h': [],
 'entities__media__0__sizes__medium__resize': [],
 'entities__media__0__sizes__medium__w': [],
 'entities__media__0__sizes__small__h': [],
 'entities__media__0__sizes__small__resize': [],
 'entities__media__0__sizes__small__w': [],
 'entities__media__0__sizes__thumb__h': [],
 'entities__media__0__sizes__thumb__resize': [],
 'entities__media__0__sizes__thumb__w': [],
 'entities__media__0__source_status_id': [],
 'entities__media__0__source_status_id_str': [],
 'entities__media__0__source_user_id': [],
 'entities__media__0__source_user_id_str': [],
 'entities__media__0__type': [],
 'entities__media__0__url': [],
 'entities__user_mentions__0__id': [],
 'entities__user_mentions__0__id_str': [],
 'entities__user_mentions__0__indices__0': [],
 'entities__user_mentions__0__indices__1': [],
 'entities__user_mentions__0__name': [],
 'entities__user_mentions__0__screen_name': [],
 'extended_entities__media__0__display_url': [],
 'extended_entities__media__0__expanded_url': [],
 'extended_entities__media__0__id': [],
 'extended_entities__media__0__id_str': [],
 'extended_entities__media__0__indices__0': [],
 'extended_entities__media__0__indices__1': [],
 'extended_entities__media__0__media_url': [],
 'extended_entities__media__0__media_url_https': [],
 'extended_entities__media__0__sizes__large__h': [],
 'extended_entities__media__0__sizes__large__resize': [],
 'extended_entities__media__0__sizes__large__w': [],
 'extended_entities__media__0__sizes__medium__h': [],
 'extended_entities__media__0__sizes__medium__resize': [],
 'extended_entities__media__0__sizes__medium__w': [],
 'extended_entities__media__0__sizes__small__h': [],
 'extended_entities__media__0__sizes__small__resize': [],
 'extended_entities__media__0__sizes__small__w': [],
 'extended_entities__media__0__sizes__thumb__h': [],
 'extended_entities__media__0__sizes__thumb__resize': [],
 'extended_entities__media__0__sizes__thumb__w': [],
 'extended_entities__media__0__source_status_id': [],
 'extended_entities__media__0__source_status_id_str': [],
 'extended_entities__media__0__source_user_id': [],
 'extended_entities__media__0__source_user_id_str': [],
 'extended_entities__media__0__type': [],
 'extended_entities__media__0__url': [],
 'favorite_count': [],
 'favorited': [],
 'geo': [],
 'id': [],
 'id_str': [],
 'in_reply_to_screen_name': [],
 'in_reply_to_status_id': [],
 'in_reply_to_status_id_str': [],
 'in_reply_to_user_id': [],
 'in_reply_to_user_id_str': [],
 'is_quote_status': [],
 'lang': [],
 'metadata__iso_language_code': [],
 'metadata__result_type': [],
 'place': [],
 'possibly_sensitive': [],
 'retweet_count': [],
 'retweeted': [],
 'retweeted_status__contributors': [],
 'retweeted_status__coordinates': [],
 'retweeted_status__created_at': [],
 'retweeted_status__entities__media__0__display_url': [],
 'retweeted_status__entities__media__0__expanded_url': [],
 'retweeted_status__entities__media__0__id': [],
 'retweeted_status__entities__media__0__id_str': [],
 'retweeted_status__entities__media__0__indices__0': [],
 'retweeted_status__entities__media__0__indices__1': [],
 'retweeted_status__entities__media__0__media_url': [],
 'retweeted_status__entities__media__0__media_url_https': [],
 'retweeted_status__entities__media__0__sizes__large__h': [],
 'retweeted_status__entities__media__0__sizes__large__resize': [],
 'retweeted_status__entities__media__0__sizes__large__w': [],
 'retweeted_status__entities__media__0__sizes__medium__h': [],
 'retweeted_status__entities__media__0__sizes__medium__resize': [],
 'retweeted_status__entities__media__0__sizes__medium__w': [],
 'retweeted_status__entities__media__0__sizes__small__h': [],
 'retweeted_status__entities__media__0__sizes__small__resize': [],
 'retweeted_status__entities__media__0__sizes__small__w': [],
 'retweeted_status__entities__media__0__sizes__thumb__h': [],
 'retweeted_status__entities__media__0__sizes__thumb__resize': [],
 'retweeted_status__entities__media__0__sizes__thumb__w': [],
 'retweeted_status__entities__media__0__type': [],
 'retweeted_status__entities__media__0__url': [],
 'retweeted_status__extended_entities__media__0__display_url': [],
 'retweeted_status__extended_entities__media__0__expanded_url': [],
 'retweeted_status__extended_entities__media__0__id': [],
 'retweeted_status__extended_entities__media__0__id_str': [],
 'retweeted_status__extended_entities__media__0__indices__0': [],
 'retweeted_status__extended_entities__media__0__indices__1': [],
 'retweeted_status__extended_entities__media__0__media_url': [],
 'retweeted_status__extended_entities__media__0__media_url_https': [],
 'retweeted_status__extended_entities__media__0__sizes__large__h': [],
 'retweeted_status__extended_entities__media__0__sizes__large__resize': [],
 'retweeted_status__extended_entities__media__0__sizes__large__w': [],
 'retweeted_status__extended_entities__media__0__sizes__medium__h': [],
 'retweeted_status__extended_entities__media__0__sizes__medium__resize': [],
 'retweeted_status__extended_entities__media__0__sizes__medium__w': [],
 'retweeted_status__extended_entities__media__0__sizes__small__h': [],
 'retweeted_status__extended_entities__media__0__sizes__small__resize': [],
 'retweeted_status__extended_entities__media__0__sizes__small__w': [],
 'retweeted_status__extended_entities__media__0__sizes__thumb__h': [],
 'retweeted_status__extended_entities__media__0__sizes__thumb__resize': [],
 'retweeted_status__extended_entities__media__0__sizes__thumb__w': [],
 'retweeted_status__extended_entities__media__0__type': [],
 'retweeted_status__extended_entities__media__0__url': [],
 'retweeted_status__favorite_count': [],
 'retweeted_status__favorited': [],
 'retweeted_status__geo': [],
 'retweeted_status__id': [],
 'retweeted_status__id_str': [],
 'retweeted_status__in_reply_to_screen_name': [],
 'retweeted_status__in_reply_to_status_id': [],
 'retweeted_status__in_reply_to_status_id_str': [],
 'retweeted_status__in_reply_to_user_id': [],
 'retweeted_status__in_reply_to_user_id_str': [],
 'retweeted_status__is_quote_status': [],
 'retweeted_status__lang': [],
 'retweeted_status__metadata__iso_language_code': [],
 'retweeted_status__metadata__result_type': [],
 'retweeted_status__place': [],
 'retweeted_status__possibly_sensitive': [],
 'retweeted_status__retweet_count': [],
 'retweeted_status__retweeted': [],
 'retweeted_status__source': [],
 'retweeted_status__text': [],
 'retweeted_status__truncated': [],
 'retweeted_status__user__contributors_enabled': [],
 'retweeted_status__user__created_at': [],
 'retweeted_status__user__default_profile': [],
 'retweeted_status__user__default_profile_image': [],
 'retweeted_status__user__description': [],
 'retweeted_status__user__favourites_count': [],
 'retweeted_status__user__follow_request_sent': [],
 'retweeted_status__user__followers_count': [],
 'retweeted_status__user__following': [],
 'retweeted_status__user__friends_count': [],
 'retweeted_status__user__geo_enabled': [],
 'retweeted_status__user__has_extended_profile': [],
 'retweeted_status__user__id': [],
 'retweeted_status__user__id_str': [],
 'retweeted_status__user__is_translation_enabled': [],
 'retweeted_status__user__is_translator': [],
 'retweeted_status__user__lang': [],
 'retweeted_status__user__listed_count': [],
 'retweeted_status__user__location': [],
 'retweeted_status__user__name': [],
 'retweeted_status__user__notifications': [],
 'retweeted_status__user__profile_background_color': [],
 'retweeted_status__user__profile_background_image_url': [],
 'retweeted_status__user__profile_background_image_url_https': [],
 'retweeted_status__user__profile_background_tile': [],
 'retweeted_status__user__profile_banner_url': [],
 'retweeted_status__user__profile_image_url': [],
 'retweeted_status__user__profile_image_url_https': [],
 'retweeted_status__user__profile_link_color': [],
 'retweeted_status__user__profile_sidebar_border_color': [],
 'retweeted_status__user__profile_sidebar_fill_color': [],
 'retweeted_status__user__profile_text_color': [],
 'retweeted_status__user__profile_use_background_image': [],
 'retweeted_status__user__protected': [],
 'retweeted_status__user__screen_name': [],
 'retweeted_status__user__statuses_count': [],
 'retweeted_status__user__time_zone': [],
 'retweeted_status__user__translator_type': [],
 'retweeted_status__user__url': [],
 'retweeted_status__user__utc_offset': [],
 'retweeted_status__user__verified': [],
 'source': [],
 'text': [],
 'truncated': [],
 'user__contributors_enabled': [],
 'user__created_at': [],
 'user__default_profile': [],
 'user__default_profile_image': [],
 'user__description': [],
 'user__favourites_count': [],
 'user__follow_request_sent': [],
 'user__followers_count': [],
 'user__following': [],
 'user__friends_count': [],
 'user__geo_enabled': [],
 'user__has_extended_profile': [],
 'user__id': [],
 'user__id_str': [],
 'user__is_translation_enabled': [],
 'user__is_translator': [],
 'user__lang': [],
 'user__listed_count': [],
 'user__location': [],
 'user__name': [],
 'user__notifications': [],
 'user__profile_background_color': [],
 'user__profile_background_image_url': [],
 'user__profile_background_image_url_https': [],
 'user__profile_background_tile': [],
 'user__profile_banner_url': [],
 'user__profile_image_url': [],
 'user__profile_image_url_https': [],
 'user__profile_link_color': [],
 'user__profile_sidebar_border_color': [],
 'user__profile_sidebar_fill_color': [],
 'user__profile_text_color': [],
 'user__profile_use_background_image': [],
 'user__protected': [],
 'user__screen_name': [],
 'user__statuses_count': [],
 'user__time_zone': [],
 'user__translator_type': [],
 'user__url': [],
 'user__utc_offset': [],
 'user__verified': []}



def research_mot(mot_clés):
  id = 10000000000000000000000000000000000000
  k = 100
  t = 0
  while(k>=100):
    k=0
    time.sleep(10)
    for stat in twitter.search(q=mot_clés, count=100, max_id=id-1)["statuses"]:
        status = flatten(stat)
        k+=1
        if(status["id"] < id):
            id = status["id"]
        for key in dict_tweet:
            try:
                dict_tweet[key].append(status[key])
            except:
                dict_tweet[key].append(None)
    t += k
    print(t)
  print(t)

def research_liste(liste1, liste2):
    t = 0
    for first in liste1:
        for second in liste2:
         id = 10000000000000000000000000000000000000
         k = 100
         while (k >= 99):
            k = 0
            time.sleep(5)
            for stat in twitter.search(q=[first, second], count=100, max_id=id - 1, lang = "fr")["statuses"]:
                status = flatten(stat)
                k += 1
                if (status["id"] < id):
                    id = status["id"]
                for key in dict_tweet:
                    try:
                        dict_tweet[key].append(status[key])
                    except:
                        dict_tweet[key].append(None)
            t += k
    print(t)

#research_liste(["controleur"], ["ratp", "metro", "rer"])

research_mot("controleur")
# "alerte", "attention", "agent", "amende"
#research_liste(["controle"], ["ratp", "attention", "couloir", "guichet", "tourniquet", "couloir", "train", "metro", "rer", "tram", "ligne", "rerA", "rerB", "rerB", "rerC", "rerD", "rerE", "rerF", "ligne1", "ligne2", "ligne3", "ligne4", "ligne5", "ligne6", "ligne7", "ligne8", "ligne9", "ligne10", "ligne11", "ligne12", "ligne13", "ligne14", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10", "M11", "M12", "M13", "M14"])

#research_liste(["controle"], ["ratp", "attention", "metro", "chatlet", "chatelet", "guichet", "tourniquet", "couloir", "Montrouge","Marcadet Poissonniers","Duvernet","Pantin","Richard Lenoir","Breguet Sabin","Oberkampf","Simplon","Aubervilliers","Pantin","Pont Neuf","Pre Saint Gervais","Choisy","Italie","Louis Blanc","Ivry","Botzaris","Jussieu","Crimee","Monge","Villejuif","LagrangeGare","Denfert","Rochereau","Chevaleret","Trocadero","Raspail","Picpus","Passy","Daumesnil","Montparnasse","Nord","Quai","Richard","Lenoir","Boissiere","Corvisart","Dugommier","Place","Glaciere","Bercy","Champerret","Reaumur","Sebastopol","Arts","Metiers","Anatole","Lachaise","Malesherbes","Raspail","Pereire","Bienvenue","Est","Clignancourt","Halles","Odeon","Vavin","Pompe","Michel","Molitor","Ambroise","Miromesnil","Creteil","Daumesnil","Saint Denis","Charenton","Reuilly","Diderot","Doree","Republique"])

#research_liste("attention", ["ratp", "chatlet", "chatelet", "guichet", "tourniquet", "couloir", "Montrouge","Marcadet Poissonniers","Duvernet","Pantin","Richard Lenoir","Breguet Sabin","Oberkampf","Simplon","Aubervilliers","Pantin","Pont Neuf","Pre Saint Gervais","Choisy","Italie","Louis Blanc","Ivry","Botzaris","Jussieu","Crimee","Monge","Villejuif","LagrangeGare","Denfert","Rochereau","Chevaleret","Trocadero","Raspail","Picpus","Passy","Daumesnil","Montparnasse","Nord","Quai","Richard","Lenoir","Boissiere","Corvisart","Dugommier","Place","Glaciere","Bercy","Champerret","Reaumur","Sebastopol","Arts","Metiers","Anatole","Lachaise","Malesherbes","Raspail","Pereire","Bienvenue","Est","Clignancourt","Halles","Odeon","Vavin","Pompe","Michel","Molitor","Ambroise","Miromesnil","Creteil","Daumesnil","Saint Denis","Charenton","Reuilly","Diderot","Doree","Republique"])

#research_liste("amende", ["ratp", "attention","chatlet", "chatelet", "guichet", "tourniquet", "couloir", "Montrouge","Marcadet Poissonniers","Duvernet","Pantin","Richard Lenoir","Breguet Sabin","Oberkampf","Simplon","Aubervilliers","Pantin","Pont Neuf","Pre Saint Gervais","Choisy","Italie","Louis Blanc","Ivry","Botzaris","Jussieu","Crimee","Monge","Villejuif","LagrangeGare","Denfert","Rochereau","Chevaleret","Trocadero","Raspail","Picpus","Passy","Daumesnil","Montparnasse","Nord","Quai","Richard","Lenoir","Boissiere","Corvisart","Dugommier","Place","Glaciere","Bercy","Champerret","Reaumur","Sebastopol","Arts","Metiers","Anatole","Lachaise","Malesherbes","Raspail","Pereire","Bienvenue","Est","Clignancourt","Halles","Odeon","Vavin","Pompe","Michel","Molitor","Ambroise","Miromesnil","Creteil","Daumesnil","Saint Denis","Charenton","Reuilly","Diderot","Doree","Republique"])

#research_liste("warning", ["ratp", "attention","chatlet", "chatelet", "guichet", "tourniquet", "couloir", "Montrouge","Marcadet Poissonniers","Duvernet","Pantin","Richard Lenoir","Breguet Sabin","Oberkampf","Simplon","Aubervilliers","Pantin","Pont Neuf","Pre Saint Gervais","Choisy","Italie","Louis Blanc","Ivry","Botzaris","Jussieu","Crimee","Monge","Villejuif","LagrangeGare","Denfert","Rochereau","Chevaleret","Trocadero","Raspail","Picpus","Passy","Daumesnil","Montparnasse","Nord","Quai","Richard","Lenoir","Boissiere","Corvisart","Dugommier","Place","Glaciere","Bercy","Champerret","Reaumur","Sebastopol","Arts","Metiers","Anatole","Lachaise","Malesherbes","Raspail","Pereire","Bienvenue","Est","Clignancourt","Halles","Odeon","Vavin","Pompe","Michel","Molitor","Ambroise","Miromesnil","Creteil","Daumesnil","Saint Denis","Charenton","Reuilly","Diderot","Doree","Republique"])

#research_liste("control", ["ratp", "attention","chatlet", "chatelet", "guichet", "tourniquet", "couloir", "Montrouge","Marcadet Poissonniers","Duvernet","Pantin","Richard Lenoir","Breguet Sabin","Oberkampf","Simplon","Aubervilliers","Pantin","Pont Neuf","Pre Saint Gervais","Choisy","Italie","Louis Blanc","Ivry","Botzaris","Jussieu","Crimee","Monge","Villejuif","LagrangeGare","Denfert","Rochereau","Chevaleret","Trocadero","Raspail","Picpus","Passy","Daumesnil","Montparnasse","Nord","Quai","Richard","Lenoir","Boissiere","Corvisart","Dugommier","Place","Glaciere","Bercy","Champerret","Reaumur","Sebastopol","Arts","Metiers","Anatole","Lachaise","Malesherbes","Raspail","Pereire","Bienvenue","Est","Clignancourt","Halles","Odeon","Vavin","Pompe","Michel","Molitor","Ambroise","Miromesnil","Creteil","Daumesnil","Saint Denis","Charenton","Reuilly","Diderot","Doree","Republique"])

#["chatlet", "chatelet", "guichet", "tourniquet", "couloir", "train", "metro", "rer", "tram", "ligne", "rerA", "rerB", "rerB", "rerC", "rerD", "rerE", "rerF", "ligne1", "ligne2", "ligne3", "ligne4", "ligne5", "ligne6", "ligne7", "ligne8", "ligne9", "ligne10", "ligne11", "ligne12", "ligne13", "ligne14", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10", "M11", "M12", "M13", "M14"]




df1 = pd.DataFrame.from_dict(dict_tweet, orient='index')
df1 = pd.DataFrame.transpose(df1)
df1.shape
df1.to_csv("controle_semaine4.csv", header=True)
