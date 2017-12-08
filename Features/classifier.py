from lexical import *

def classifier(url):
    raw_features=[]

    # Content features tuple
    raw_features.append(
         #Lexical features tuple.
        (
        url_length(url),
        special_chars(url),
        ratio_special_chars(url),
        token_count(url),
        Presence_of_IP(url),
        # getTokens(url),                      # Returns array of words
        suspicious_word_count(url),
        # domain_name(url),                    # Returns string
        # subdomain_name(url),                 # Returns string
        subdomain_length(url),
        domain_token_count(url),
        # longest_domain_token_count(url),     # Returns string
        query_variables_count(url),
        max_length_variable(url),
        countdelim(url),
        countSubDomain(url),
        alphabet_count(url),
        digit_count(url),
        countQueries(url),
        countdots(url),
        count_at_symbol(url),
        argument_length(url),
        isPresentHyphen(url),
        isPresentAt(url),
        countSubDir(url),
        # get_ext(url),                        # Returns string
        # get_filename(url),                   # Returns string
        # URL_path(url),                       # Returns string
        # URL_scheme(url),                     # Returns string
        scheme_http_or_not(url),
        path_length(url),
        directory_length(url),
        # sub_directory(url),                  # Returns string
        sub_directory_special_count(url),
        sub_directory_tokens_count(url),
        # filename(url),                       # Returns string
        filename_length(url),
        port_number(url),
        blacklisted_word_present(url),
        longest_token_path(url),
        hyphens_instead_dots_domain(url),
        hostname_unicode(url),
        another_char_hostname(url)
        )
    )
    print(raw_features)

classifier("www.google.com")
