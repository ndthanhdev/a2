# -*- coding: utf-8 -*-
from keras.models import Model, load_model
import numpy as np
import re
from keras.preprocessing.sequence import pad_sequences
from gensim.models.keyedvectors import KeyedVectors
from gensim.models import Word2Vec
from babi_rnn_vi import tokenize


def main():
    print('Loading...')
    model = load_model('outputs/babi.h5')

    [word2idx, story_maxlen, query_maxlen] = np.load(
        'outputs/model_context.npy')
    idx2word = dict([(v, k) for k, v in word2idx.items()])
    word2vec = KeyedVectors.load_word2vec_format('outputs/vi.vec')
    # word2vec = KeyedVectors.load_word2vec_format('outputs/word2vec.vec')

    def mostSimilarity(externalVoca):
        return max([k for k in word2idx if k in word2vec.wv.vocab], key=lambda k: word2vec.wv.similarity(externalVoca, k))

    def toVec(words):
        vectors = []
        for w in words:
            if w not in word2idx:
                # w = mostSimilarity(w)
                print(w)
            else:
                vectors.append(word2idx[w])
        return vectors

    def toWord(idx):
        return idx2word[idx]

    def predict(corpus, query):
        corpus = tokenize("Võ Thị Sáu là nữ anh hùng, sinh năm 1933 ở huyện Đất Đỏ, tỉnh Bà Rịa. Sinh ra và lớn lên trên miền quê giàu truyền thống yêu nước, lại chứng kiến cảnh thực dân Pháp giết chóc đồng bào, chị Sáu đã không ngần ngại cùng các anh trai tham gia cách mạng. 14 tuổi, Võ Thị Sáu theo anh gia nhập Việt Minh, trốn lên chiến khu chống Pháp. Chị tham gia đội công an xung phong, hoàn thành xuất sắc nhiệm vụ liên lạc, tiếp tế. Trong khoảng thời gian này, chị Sáu tham gia nhiều trận chiến đấu để bảo vệ quê hương, dùng lựu đạn tiêu diệt hai tên ác ôn và làm bị thương nhiều lính Pháp. Người con gái Đất Đỏ còn nhiều lần phát hiện gian tế, tay sai Pháp, giúp đội công an thoát khỏi nguy hiểm, chủ động tấn công địch. Tháng 7/1948, Công an Đất Đỏ được giao nhiệm vụ phá cuộc mít tinh kỷ niệm Quốc khánh Pháp. Biết đây là nhiệm vụ gian nan, nguy hiểm, chị Sáu vẫn chủ động xin được trực tiếp đánh trận này. Chị nhận lựu đạn, ém vào góc chợ gần khán đài từ nửa đêm. Sáng hôm đó, địch lùa người dân vào sân.    Khi xe của tỉnh trưởng tới, chị tung lựu đạn về phía khán đài, uy hiếp giải tán mít tinh. Hai tổ công an xung phong chốt gần đấy đồng loạt nổ súng yểm trợ tạo áp lực giải tán cuộc mít tinh, đồng thời hỗ trợ cho chị Sáu rút an toàn. Người của Việt Minh được bố trí trong đám đông hô to \"Việt Minh tiến công\" và hướng dẫn người dân giải tán. Sau chiến công này, chị Sáu được tổ chức tuyên dương khen ngợi và được giao nhiệm vụ diệt tề trừ gian, bao gồm việc tiêu diệt tên cai tổng Tòng. Tháng 11/1948, Võ Thị Sáu mang theo lựu đạn, trà trộn vào đám người đi làm căn cước. Giữa buổi, chị ném lựu đạn vào nơi làm việc của Tòng, hô to \"Việt Minh tấn công\" rồi kéo mấy chị em cùng chạy. Lựu đạn nổ, tên Tòng bị thương nặng nhưng không chết. Tuy nhiên, vụ tấn công khiến bọn lính đồn khiếp vía, không dám truy lùng Việt Minh ráo riết như trước. 2/1950, Võ Thị Sáu tiếp tục nhận nhiệm vụ ném lựu đạn, tiêu diệt hai chỉ điểm viên của thực dân Pháp là Cả Suốt và Cả Đay rồi không may bị bắt. Trong hơn một tháng bị giam tại nhà tù Đất Đỏ, dù bị giặc tra tấn dã man, chị không khai báo. Địch phải chuyển chị về khám Chí Hòa. Chị Sáu tiếp tục làm liên lạc cho các đồng chí trong khám, cùng chị em tại tù đấu tranh đòi cải thiện cuộc sống nhà tù. Trước tinh thần đấu tranh quyết liệt của Võ Thị Sáu, thực dân Pháp và tay sai mở phiên tòa, kết án tử hình đối với nữ chiến sĩ trẻ. Chúng chuyển chị cùng một số người tù cách mạng ra nhà tù Côn Đảo. Nhờ sự kiên cường, dũng cảm, trung thành, Võ Thị Sáu được kết nạp vào Đảng Lao động Việt Nam và công nhận là Đảng viên chính thức ngày đêm trước khi hy sinh. Trong quá trình bị bắt, tra tấn và đến tận những giây phút cuối cùng, Võ Thị Sáu luôn chứng tỏ bản lĩnh kiên cường, bất khuất của chiến sĩ cộng sản. Khi mới bị bắt, địch tra tấn chị chết đi sống lại nhưng không moi được nửa lời khai báo. Sự kiên trung ấy một lần nữa thể hiện tại phiên tòa đại hình khi chị Sáu(khi đó mới 17 tuổi) hiên ngang khẳng định: \"Yêu nước, chống bọn thực dân xâm lược không phải là tội\". Khi nhận án tử hình, chị Sáu không hề run sợ. Chị hô to \"Đả đảo thực dân Pháp!\", \"Kháng chiến nhất định thắng lợi!\". Năm 1952, trước giờ hành hình, viên cha đạo đề nghị làm lễ rửa rội cho chị. Song chị từ chối và nói: \"Tôi không có tội. Chỉ có kẻ sắp hành hình tôi đây mới có tội\". Đối mặt cái chết, điều khiến người con gái Đất Đỏ ân hận nhất là chưa diệt hết bọn thực dân và tay sai cướp nước. Giai thoại kể rằng khi ra đến pháp trường, Võ Thị Sáu kiên quyết không quỳ xuống, yêu cầu không bịt mắt. \"Không cần bịt mắt tôi. Hãy để cho đôi mắt tôi được nhìn đất nước thân yêu đến giây phút cuối cùng và tôi có đủ can đảm để nhìn thẳng vào họng súng của các người!\", chị tuyên bố. Nói xong, chị Sáu bắt đầu hát Tiến quân ca. Khi lính lên đạn, chị ngừng hát, hô vang những lời cuối cùng \"Đả đảo bọn thực dân Pháp. Việt Nam độc lập muôn năm. Hồ Chủ tịch muôn năm!\".")
        query = tokenize(query)
        # print('corpus', corpus)
        print('query', query)
        input = [pad_sequences([toVec(corpus)], story_maxlen), pad_sequences(
            [toVec(query)], query_maxlen)]
        output = model.predict(input, 32)
        return toWord(np.argmax(output))

    corpus = []
    print('Chào bạn!')
    while True:
        temp = str(input('You: '))
        if temp == 'exit':
            print('Bot: Tạm biệt')
            exit()
        elif temp == 'new':
            corpus = []
            print('Bot: Quên hết rồi.')
        elif '?' in temp:
            # print('Bot: ', predict('. '.join(corpus), temp))
            print('Bot: ', predict("", temp))
        else:
            corpus.append(temp.strip())
            print('Bot: ờ')


if __name__ == '__main__':
    main()
