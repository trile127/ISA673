from lexical import *

def classifier(url):
    # Lexical feature array
    raw_features=[                          # Feature Number
        url_length(url),                    # F01
        special_chars(url),                 # F02
        ratio_special_chars(url),           # F03
        token_count(url),                   # F04
        Presence_of_IP(url),                # F05
        suspicious_word_count(url),         # F06
        subdomain_length(url),              # F07
        domain_token_count(url),            # F08
        query_variables_count(url),         # F09
        max_length_variable(url),           # F10
	max_length_variable_value(url),     # F11
        countdelim(url),                    # F12
        countSubDomain(url),                # F13
        alphabet_count(url),                # F14
        digit_count(url),                   # F15
        countQueries(url),                  # F16
        countdots(url),                     # F17
        count_at_symbol(url),               # F18
        argument_length(url),               # F19
        isPresentHyphen(url),               # F20
        isPresentAt(url),                   # F21
        countSubDir(url),                   # F22
        scheme_http_or_not(url),            # F23
        path_length(url),                   # F24
        directory_length(url),              # F25
        sub_directory_special_count(url),   # F26
        sub_directory_tokens_count(url),    # F27
        filename_length(url),               # F28
        port_number(url),                   # F29
        blacklisted_word_present(url),      # F30
        longest_token_path(url),            # F31
        hyphens_instead_dots_domain(url),   # F32
        exe_in_url(url)                     # F33
    ]
    return raw_features

# Example
# print(classifier("www.google.com"))
